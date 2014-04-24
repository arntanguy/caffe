// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    test_net net_proto pretrained_net_proto iterations [CPU/GPU]

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

template <typename Dtype>
class VeriSGDSolver : public Solver<Dtype> {
	public:
		explicit VeriSGDSolver(const SolverParameter& param, const string &input2)
			: Solver<Dtype>(param) {
				/* build shadow */
				LOG(INFO) << "BUILD SHADOW NET";
				DATA_LAYER_IDX = 0;
				NetParameter train_net_param;
				ReadProtoFromTextFile(param.train_net(), &train_net_param);
				LOG(INFO) << "INPUT1 " << train_net_param.layers(DATA_LAYER_IDX).layer().source();
				LOG(INFO) << "INPUT2 " << input2;
				train_net_param.mutable_layers(DATA_LAYER_IDX)->mutable_layer()->set_source_list(input2);
				train_net_param.mutable_layers(DATA_LAYER_IDX)->mutable_layer()->set_share_data(true);

				shadow_net_.reset(new Net<Dtype>(train_net_param));
				const ShuffleDataLayer<Dtype> *data_layer = dynamic_cast<ShuffleDataLayer<Dtype> *>(this->net_->layers()[DATA_LAYER_IDX].get());
				ShuffleDataLayer<Dtype> *shadow_data_layer = dynamic_cast<ShuffleDataLayer<Dtype> *>(this->shadow_net_->mutable_layers()[DATA_LAYER_IDX].get());
				CHECK(data_layer != NULL);
				shadow_data_layer->CopyDataPtrFrom(*data_layer);

				feature_layer_id = -1;
				for(int i=0;i<this->net_->layers().size();i++){
					if(this->net_->layer_names()[i] == "relu5"){
						feature_layer_id = i;
						break;
					}
				}
				CHECK_GE(feature_layer_id, 0);
				LOG(INFO) << "feature layer id " << feature_layer_id;

				//Setup loss layer
				vector<Blob<Dtype>*> & feat_vec1 = this->net_->top_vecs()[feature_layer_id];
				vector<Blob<Dtype>*> & feat_vec2 = this->shadow_net_->top_vecs()[feature_layer_id];
				CHECK_EQ(feat_vec1.size(), 1);
				CHECK_EQ(feat_vec2.size(), 1);
				TOP_LAYER_ID = this->net_->top_vecs().size() - 1;
				CHECK_EQ(this->net_->top_vecs()[DATA_LAYER_IDX].size(), 2);
				CHECK_EQ(this->net_->top_vecs()[TOP_LAYER_ID].size(), 0);

				Blob<Dtype>* & label_vec1 = this->net_->top_vecs()[DATA_LAYER_IDX][1];
				Blob<Dtype>* & label_vec2 = this->shadow_net_->top_vecs()[DATA_LAYER_IDX][1];

				//Blob<Dtype>* & top_vec1 = this->net_->top_vecs()[TOP_LAYER_ID][0];
				//Blob<Dtype>* & top_vec2 = this->shadow_net_->top_vecs()[TOP_LAYER_ID][0];

				LayerParameter __dummy_lp;
				loss_layer.reset(new VerificationLossLayer<Dtype>(__dummy_lp));

				loss_bottom.push_back(feat_vec1[0]);
				loss_bottom.push_back(label_vec1);
				loss_bottom.push_back(feat_vec2[0]);
				loss_bottom.push_back(label_vec2);

				loss_layer->SetUp(loss_bottom, &loss_top);

				loss_layer->ALPHA = Dtype(0.1);
				loss_layer->M = Dtype(16);
				LOG(INFO) << "BUILD LOSS DONE";

			}

		/* overload, hide Solve in base class */
		void Solve(const char* resume_file);
	protected:
		virtual void PreSolve();
		Dtype GetLearningRate();
		virtual void ComputeUpdateValue();
		virtual void SnapshotSolverState(SolverState * state);
		virtual void RestoreSolverState(const SolverState& state);

		void SyncNet();
		// history maintains the historical momentum data.
		vector<shared_ptr<Blob<Dtype> > > history_;

		shared_ptr<Net<Dtype> > shadow_net_;

		shared_ptr<VerificationLossLayer<Dtype> > loss_layer;
		int feature_layer_id;
		int DATA_LAYER_IDX;
		int TOP_LAYER_ID;
		vector<Blob<Dtype>*> loss_bottom;
		vector<Blob<Dtype>*> loss_top;

		DISABLE_COPY_AND_ASSIGN(VeriSGDSolver);
};

template <typename Dtype>
void VeriSGDSolver<Dtype>::SyncNet() {
	shadow_net_->CopyLayersFrom(*this->net_, false);
}

template <typename Dtype>
void VeriSGDSolver<Dtype>::Solve(const char* resume_file) {
	Caffe::set_mode(Caffe::Brew(this->param_.solver_mode()));
	if (this->param_.solver_mode() && this->param_.has_device_id()) {
		Caffe::SetDevice(this->param_.device_id());
	}
	Caffe::set_phase(Caffe::TRAIN);
	LOG(INFO) << "Solving(DUAL) " << this->net_->name();
	PreSolve();

	this->iter_ = 0;
	if (resume_file) {
		LOG(INFO) << "Restoring previous solver status from " << resume_file;
		this->Restore(resume_file);
	}

	CHECK_EQ(this->net_->bottom_vecs()[feature_layer_id+1][0]->cpu_diff(), 
		loss_bottom[0]->cpu_diff());

	// For a network that is trained by the solver, no bottom or top vecs
	// should be given, and we will just provide dummy vecs.
	vector<Blob<Dtype>*> bottom_vec;
	while (this->iter_++ < this->param_.max_iter()) {
		//Dtype loss = this->net_->ForwardBackward(bottom_vec);
		Dtype loss = 0.;
		SyncNet();
		this->net_->Forward(bottom_vec);
		shadow_net_->Forward(bottom_vec);

		loss_layer->Forward(loss_bottom, &loss_top);
		//BP
		//get diff
		loss += this->net_->BackwardBetween(TOP_LAYER_ID, feature_layer_id+1);
		loss += this->shadow_net_->BackwardBetween(TOP_LAYER_ID, feature_layer_id+1);
		//gradient add to bottom layer
		loss += loss_layer->Backward(loss_top, true, &loss_bottom);

		loss += this->net_->BackwardBetween(feature_layer_id, 0);
		loss += this->shadow_net_->BackwardBetween(feature_layer_id, 0);

		//XXX
		ComputeUpdateValue();
		this->net_->Update();

		if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
			LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << loss;
		}
		if (this->param_.test_interval() && this->iter_ % this->param_.test_interval() == 0) {
			// We need to set phase to test before running.
			Caffe::set_phase(Caffe::TEST);
			this->Test();
			Caffe::set_phase(Caffe::TRAIN);
		}
		// Check if we need to do snapshot
		if (this->param_.snapshot() && this->iter_ % this->param_.snapshot() == 0) {
			this->Snapshot();
		}
	}
	// After the optimization is done, always do a snapshot.
	this->iter_--;
	this->Snapshot();
	LOG(INFO) << "Optimization Done.";
}

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
// where base_lr, gamma, step and power are defined in the solver parameter
// protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype VeriSGDSolver<Dtype>::GetLearningRate() {
	Dtype rate;
	const string& lr_policy = this->param_.lr_policy();
	if (lr_policy == "fixed") {
		rate = this->param_.base_lr();
	} else if (lr_policy == "step") {
		int current_step = this->iter_ / this->param_.stepsize();
		rate = this->param_.base_lr() *
			pow(this->param_.gamma(), current_step);
	} else if (lr_policy == "exp") {
		rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
	} else if (lr_policy == "inv") {
		rate = this->param_.base_lr() *
			pow(Dtype(1) + this->param_.gamma() * this->iter_,
					- this->param_.power());
	} else {
		LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
	}
	return rate;
}


template <typename Dtype>
void VeriSGDSolver<Dtype>::PreSolve() {
	// Initialize the history
	vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
	history_.clear();
	LOG(INFO) << "VeriSGDSolver::PreSolve";
	for (int i = 0; i < net_params.size(); ++i) {
		const Blob<Dtype>* net_param = net_params[i].get();
		history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
						net_param->num(), net_param->channels(), net_param->height(),
						net_param->width())));
	}
}


template <typename Dtype>
void VeriSGDSolver<Dtype>::ComputeUpdateValue() {
	vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
	vector<float>& net_params_lr = this->net_->params_lr();
	vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
	// get the learning rate
	Dtype rate = GetLearningRate();
	if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
		LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
	}
	Dtype momentum = this->param_.momentum();
	Dtype weight_decay = this->param_.weight_decay();
	switch (Caffe::mode()) {
		case Caffe::CPU:
			for (int param_id = 0; param_id < net_params.size(); ++param_id) {
				// Compute the value to history, and then copy them to the blob's diff.
				Dtype local_rate = rate * net_params_lr[param_id];
				Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
				//add diff from net2 to net1
				caffe_axpy(net_params[param_id]->count(), Dtype(1.),
						this->shadow_net_->params()[param_id]->cpu_diff(),
						net_params[param_id]->mutable_cpu_diff());

				caffe_axpby(net_params[param_id]->count(), local_rate,
						net_params[param_id]->cpu_diff(), momentum,
						history_[param_id]->mutable_cpu_data());
				if (local_decay) {
					// add weight decay
					caffe_axpy(net_params[param_id]->count(),
							local_decay * local_rate,
							net_params[param_id]->cpu_data(),
							history_[param_id]->mutable_cpu_data());
				}
				// copy
				caffe_copy(net_params[param_id]->count(),
						history_[param_id]->cpu_data(),
						net_params[param_id]->mutable_cpu_diff());
			}
			break;
		case Caffe::GPU:
			for (int param_id = 0; param_id < net_params.size(); ++param_id) {
				// Compute the value to history, and then copy them to the blob's diff.
				Dtype local_rate = rate * net_params_lr[param_id];
				Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
				//add diff from net2 to net1
				caffe_gpu_axpy(net_params[param_id]->count(), Dtype(1.),
						this->shadow_net_->params()[param_id]->gpu_diff(),
						net_params[param_id]->mutable_gpu_diff());

				caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
						net_params[param_id]->gpu_diff(), momentum,
						history_[param_id]->mutable_gpu_data());
				if (local_decay) {
					// add weight decay
					caffe_gpu_axpy(net_params[param_id]->count(),
							local_decay * local_rate,
							net_params[param_id]->gpu_data(),
							history_[param_id]->mutable_gpu_data());
				}
				// copy
				caffe_gpu_copy(net_params[param_id]->count(),
						history_[param_id]->gpu_data(),
						net_params[param_id]->mutable_gpu_diff());
			}
			break;
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
	}
}

template <typename Dtype>
void VeriSGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
	state->clear_history();
	for (int i = 0; i < history_.size(); ++i) {
		// Add history
		BlobProto* history_blob = state->add_history();
		history_[i]->ToProto(history_blob);
	}
}

template <typename Dtype>
void VeriSGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
	CHECK_EQ(state.history_size(), history_.size())
		<< "Incorrect length of history blobs.";
	LOG(INFO) << "VeriSGDSolver: restoring history";
	for (int i = 0; i < history_.size(); ++i) {
		history_[i]->FromProto(state.history(i));
	}
}


int main(int argc, char** argv) {
	if (argc < 3) {
		LOG(ERROR) << "argc";
		return -1;
	}

	//cudaSetDevice(0);
	//Caffe::set_phase(Caffe::TEST);
	Caffe::set_mode(Caffe::GPU);

	SolverParameter solver_param;
	ReadProtoFromTextFile(argv[1], &solver_param);
	VeriSGDSolver<float> solver(solver_param, argv[2]);

	LOG(INFO) << ">>> NET BUILD";

	solver.Solve(0);
	//Net<float> train_net0(cnn_train);
	//Net<float> train_net1(cnn_train);

	/*
	   NetParameter trained_net_param;
	   caffe_test_net.ToProto(&trained_net_param, false);
	   WriteProtoToBinaryFile(trained_net_param, argv[2]);
	   */
	return 0;
}

