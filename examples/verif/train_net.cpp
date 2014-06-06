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
#include <fenv.h>
#include <signal.h>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

#define GET_LAYER(var, net, id, type) type<Dtype> *var = dynamic_cast<type<Dtype> *>(net->mutable_layers()[id].get())
#define GET_CONST_LAYER(var, net, id, type) const type<Dtype> *var = dynamic_cast<const type<Dtype> *>(net->layers()[id].get())

template <typename Dtype>
class VeriSGDSolver : public Solver<Dtype> {
	public:
		explicit VeriSGDSolver(const SolverParameter& param, const LayerParameter &extra_param)
			: Solver<Dtype>(param), extra_layer_params_(extra_param) {
				/* build shadow */
				LOG(INFO) << "BUILD SHADOW NET";
				DATA_LAYER_IDX = 0;
				NetParameter train_net_param;
				ReadProtoFromTextFile(param.train_net(), &train_net_param);
				LOG(INFO) << "INPUT " << train_net_param.layers(DATA_LAYER_IDX).layer().source();
				//train_net_param.mutable_layers(DATA_LAYER_IDX)->mutable_layer()->set_source_list(input2);
				train_net_param.mutable_layers(DATA_LAYER_IDX)->mutable_layer()->set_share_data(true);

				shadow_net_.reset(new Net<Dtype>(train_net_param));

				GET_CONST_LAYER(data_layer, this->net_, DATA_LAYER_IDX, ShuffleDataLayer);
				GET_LAYER(shadow_data_layer, this->shadow_net_, DATA_LAYER_IDX, ShuffleDataLayer);

				CHECK(data_layer != NULL);
				shadow_data_layer->CopyDataPtrFrom(*data_layer);

				shadow_data_layer->SetOutputChannel(1);

				FEATURE_LAYER_ID = FindLayer("relu5");
				CHECK_GE(FEATURE_LAYER_ID, 0);
				LOG(INFO) << "feature layer id " << FEATURE_LAYER_ID;

				//Setup loss layer
				vector<Blob<Dtype>*> & feat_vec1 = this->net_->top_vecs()[FEATURE_LAYER_ID];
				vector<Blob<Dtype>*> & feat_vec2 = this->shadow_net_->top_vecs()[FEATURE_LAYER_ID];
				CHECK_EQ(feat_vec1.size(), 1);
				CHECK_EQ(feat_vec2.size(), 1);
				TOP_LAYER_ID = this->net_->layers().size() - 1;
				CHECK_EQ(this->net_->top_vecs()[DATA_LAYER_IDX].size(), 2);
				CHECK_EQ(this->net_->top_vecs()[TOP_LAYER_ID].size(), 0);

				Blob<Dtype>* & label_vec1 = this->net_->top_vecs()[DATA_LAYER_IDX][1];
				Blob<Dtype>* & label_vec2 = this->shadow_net_->top_vecs()[DATA_LAYER_IDX][1];

				//Blob<Dtype>* & top_vec1 = this->net_->top_vecs()[TOP_LAYER_ID][0];
				//Blob<Dtype>* & top_vec2 = this->shadow_net_->top_vecs()[TOP_LAYER_ID][0];

				VerificationLossLayer<Dtype> *loss = new  VerificationLossLayer<Dtype>(extra_layer_params_);
				loss->ReadCorrespondancesFile();
				loss_layer.reset(loss);

				loss_bottom.push_back(feat_vec1[0]);
				loss_bottom.push_back(label_vec1);
				loss_bottom.push_back(feat_vec2[0]);
				loss_bottom.push_back(label_vec2);

				loss_layer->SetUp(loss_bottom, &loss_top);

				LOG(INFO) << "BUILD LOSS DONE";

				LOG(INFO) << "Loading test network";
				//Build test accuracy
				NetParameter test_net_param;
				ReadProtoFromTextFile(param.test_net(), &test_net_param);
				test_net_param.mutable_layers(DATA_LAYER_IDX)->mutable_layer()->set_share_data(true);


				LOG(INFO) << "TEST_INPUT " << test_net_param.layers(DATA_LAYER_IDX).layer().source();
				shadow_test_net_.reset(new Net<Dtype>(test_net_param));

				GET_CONST_LAYER(test_data_layer, this->test_net_, DATA_LAYER_IDX, ShuffleDataLayer);
				GET_LAYER(shadow_test_data_layer, this->shadow_test_net_, DATA_LAYER_IDX, ShuffleDataLayer);

				CHECK(test_data_layer != NULL);
				shadow_test_data_layer->CopyDataPtrFrom(*test_data_layer);
				shadow_test_data_layer->SetOutputChannel(1);

				accuracy_layer.reset(new VerificationAccuracyLayer<Dtype>(extra_layer_params_));

				vector<Blob<Dtype>*> test_out_vec1 = this->test_net_->output_blobs();
				vector<Blob<Dtype>*> test_out_vec2 = this->shadow_test_net_->output_blobs();
				CHECK_EQ(test_out_vec1.size(), 2);
				CHECK_EQ(test_out_vec2.size(), 2);

				accuracy_bottom.push_back(test_out_vec1[0]);
				accuracy_bottom.push_back(test_out_vec1[1]);
				accuracy_bottom.push_back(test_out_vec2[0]);
				accuracy_bottom.push_back(test_out_vec2[1]);

				accuracy_top.push_back(&accuracy_out);

				accuracy_layer->SetUp(accuracy_bottom, &accuracy_top);
				LOG(INFO) << "BUILD ACC DONE";

				//Setup dropout
				DROPOUT_TOP_ID = FindLayer("dropout_group_2");	
				DROPOUT_BOTTOM_ID = -1;
				if(DROPOUT_TOP_ID >= 0){
					LOG(INFO) << "Setup dropout layers";
					GET_LAYER(drop2, this->net_, DROPOUT_TOP_ID, DropoutGroupLayer);
					GET_LAYER(drop2_shadow, this->shadow_net_, DROPOUT_TOP_ID, DropoutGroupLayer);
					CHECK(drop2 != NULL);
					CHECK(drop2_shadow != NULL);

					drop2_shadow->ShareMask(drop2);

					DROPOUT_BOTTOM_ID = FindLayer("dropout_group_1");	
					CHECK(DROPOUT_BOTTOM_ID >= 0);
					GET_LAYER(drop1, this->net_, DROPOUT_BOTTOM_ID, DropoutGroupLayer);
					GET_LAYER(drop1_shadow, this->shadow_net_, DROPOUT_BOTTOM_ID, DropoutGroupLayer);
					CHECK(drop1 != NULL);
					CHECK(drop1_shadow != NULL);
					drop1_shadow->ShareMask(drop1);
				}
			}

		/* overload, hide Solve in base class */
		void Solve(const char* resume_file);
	protected:
		virtual void PreSolve();
		void TestDual();
		Dtype GetLearningRate();
		virtual void ComputeUpdateValue();
		virtual void SnapshotSolverState(SolverState * state);
		virtual void RestoreSolverState(const SolverState& state);

		void SyncNet();

		int FindLayer(const std::string &name){
			int id = -1;
			for(int i=0;i<this->net_->layers().size();i++){
				if(this->net_->layer_names()[i] == name){
					id = i;
					break;
				}
			}
			return id;
		}

		// history maintains the historical momentum data.
		vector<boost::shared_ptr<Blob<Dtype> > > history_;

		boost::shared_ptr<Net<Dtype> > shadow_net_;
		boost::shared_ptr<Net<Dtype> > shadow_test_net_;

		boost::shared_ptr<VerificationLossLayer<Dtype> > loss_layer;
		boost::shared_ptr<VerificationAccuracyLayer<Dtype> > accuracy_layer;

		int FEATURE_LAYER_ID;
		int DATA_LAYER_IDX;
		int TOP_LAYER_ID;
		int DROPOUT_TOP_ID;
		int DROPOUT_BOTTOM_ID;
		vector<Blob<Dtype>*> loss_bottom;
		vector<Blob<Dtype>*> loss_top;

		vector<Blob<Dtype>*> accuracy_bottom;
		vector<Blob<Dtype>*> accuracy_top;
		Blob<Dtype> accuracy_out;
		
		LayerParameter extra_layer_params_;

		DISABLE_COPY_AND_ASSIGN(VeriSGDSolver);
};

template <typename Dtype>
void VeriSGDSolver<Dtype>::SyncNet() {
	shadow_net_->CopyLayersFrom(*this->net_, false);
}

template <typename Dtype>
void VeriSGDSolver<Dtype>::TestDual() {
	LOG(INFO) << "Iteration " << this->iter_ << ", Testing net(dual)";
	NetParameter net_param;
	this->net_->ToProto(&net_param);
	CHECK_NOTNULL(this->test_net_.get())->CopyTrainedLayersFrom(net_param);
	CHECK_NOTNULL(this->shadow_test_net_.get())->CopyTrainedLayersFrom(net_param);
	vector<Dtype> test_score;
	vector<Blob<Dtype>*> bottom_vec;

	for (int i = 0; i < this->param_.test_iter(); ++i) {
		this->test_net_->Forward(bottom_vec);
		this->shadow_test_net_->Forward(bottom_vec);
		this->accuracy_layer->Forward(accuracy_bottom, &accuracy_top);

		if (i == 0) {
			for (int j = 0; j < accuracy_top.size(); ++j) {
				const Dtype* result_vec = accuracy_top[j]->cpu_data();
				for (int k = 0; k < accuracy_top[j]->count(); ++k) {
					test_score.push_back(result_vec[k]);
				}
			}
		} else {
			int idx = 0;
			for (int j = 0; j < accuracy_top.size(); ++j) {
				const Dtype* result_vec = accuracy_top[j]->cpu_data();
				for (int k = 0; k < accuracy_top[j]->count(); ++k) {
					test_score[idx++] += result_vec[k];
				}
			}
		}
	}
	for (int i = 0; i < test_score.size(); ++i) {
		LOG(INFO) << "Test score #" << i << ": "
			<< test_score[i] / this->param_.test_iter();
	}

}


template <typename Dtype>
void VeriSGDSolver<Dtype>::Solve(const char* resume_file) {
	LOG(INFO) << "VeriSGDSolver::Solve";

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

	/*
	CHECK_EQ(this->net_->bottom_vecs()[FEATURE_LAYER_ID+1][0]->cpu_diff(), 
			loss_bottom[0]->cpu_diff());
			*/

	GET_LAYER(dropout_layer2, this->net_, std::max(DROPOUT_TOP_ID,0), DropoutGroupLayer); 
	GET_LAYER(dropout_layer1, this->net_, std::max(DROPOUT_BOTTOM_ID,0), DropoutGroupLayer); 
	bool update_dropout = dropout_layer1 != NULL;
	
	// For a network that is trained by the solver, no bottom or top vecs
	// should be given, and we will just provide dummy vecs.
	vector<Blob<Dtype>*> bottom_vec;
	while (this->iter_++ < this->param_.max_iter()) {
		//Dtype loss = this->net_->ForwardBackward(bottom_vec);

#if 1
		Dtype loss = 0., loss_v = 0.;
		SyncNet();

		if(update_dropout){
			dropout_layer2->UpdateMask();
			dropout_layer1->UpscaleMaskFrom(dropout_layer2);
		}

		this->net_->Forward(bottom_vec);
		shadow_net_->Forward(bottom_vec);

		loss_layer->Forward(loss_bottom, &loss_top);
		//BP
		//get diff
		loss += this->net_->BackwardBetween(TOP_LAYER_ID, FEATURE_LAYER_ID+1);
		loss += this->shadow_net_->BackwardBetween(TOP_LAYER_ID, FEATURE_LAYER_ID+1);
		//gradient add to bottom layer
		bool prob_vloss = this->iter_ > this->param_.pretrain_iterations();
		loss_v += loss_layer->Backward(loss_top, prob_vloss, &loss_bottom);

		loss += this->net_->BackwardBetween(FEATURE_LAYER_ID, 0);
		loss += this->shadow_net_->BackwardBetween(FEATURE_LAYER_ID, 0);
		
		loss *= 0.5;
#else
    Dtype loss = this->net_->ForwardBackward(bottom_vec);
#endif

		//XXX
		ComputeUpdateValue();
		this->net_->Update();

		if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
			vector<Dtype> means;
			Dtype thr = this->loss_layer->CalcThreshold(false);
			this->loss_layer->GetMeanDistance(means);
			LOG(INFO) << "Distance mean: " << means[0] << ", " << means[1] << ", " << thr;
			LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << loss << ", loss_v = " << loss_v;
		}
		if (this->param_.test_interval() && this->iter_ % this->param_.test_interval() == 0) {
			// We need to set phase to test before running.
			Caffe::set_phase(Caffe::TEST);
			this->TestDual();
			Caffe::set_phase(Caffe::TRAIN);
		}
		if (this->param_.update_dual_thr_interval() && this->iter_ % this->param_.update_dual_thr_interval() == 0) {
			//XXX TODO
			Dtype thr = this->loss_layer->CalcThreshold(true);
			this->accuracy_layer->SetThreshold(thr);
			LOG(INFO) << "new_thr: " << thr;
			this->loss_layer->ResetDistanceStat();
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
	vector<boost::shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
	history_.clear();
	LOG(INFO) << "VeriSGDSolver::PreSolve";
	for (int i = 0; i < net_params.size(); ++i) {
		const Blob<Dtype>* net_param = net_params[i].get();
		history_.push_back(boost::shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
						net_param->num(), net_param->channels(), net_param->height(),
						net_param->width())));
	}
}


template <typename Dtype>
void VeriSGDSolver<Dtype>::ComputeUpdateValue() {
	vector<boost::shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
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
#if 0
				LOG(INFO) << "_-------- " << param_id;
				for(int i=0;i<10;i++)
					LOG(INFO) << this->net_->params()[param_id]->cpu_diff()[i] << ' '
						<< this->shadow_net_->params()[param_id]->cpu_diff()[i] ;
#endif

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
	state->set_dual_thr(loss_layer->GetThreshold());
}

template <typename Dtype>
void VeriSGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
	CHECK_EQ(state.history_size(), history_.size())
		<< "Incorrect length of history blobs.";
	LOG(INFO) << "VeriSGDSolver: restoring history";
	for (int i = 0; i < history_.size(); ++i) {
		history_[i]->FromProto(state.history(i));
	}
	LOG(INFO) << "Restore dual_thr " << state.dual_thr();
	loss_layer->SetThreshold(state.dual_thr());
}

static void sighandler(int signum)
{
	LOG(FATAL) << "Floating point error: " << signum;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		LOG(ERROR) << "argc";
		return -1;
	}

	struct sigaction sa;
	/* trap overflow */
#if 0
	feenableexcept(FE_INVALID   | 
			FE_DIVBYZERO | 
			FE_OVERFLOW
		      );
	signal(SIGFPE, sighandler);
#endif
	//cudaSetDevice(0);
	//Caffe::set_phase(Caffe::TEST);
	//Caffe::set_mode(Caffe::GPU);

	SolverParameter solver_param;
	ReadProtoFromTextFile(argv[1], &solver_param);
	//Caffe::SetDevice(solver_param.device_id());
	LayerParameter extra_param;
	ReadProtoFromTextFile(argv[2], &extra_param);
	VeriSGDSolver<float> solver(solver_param, extra_param);

	LOG(INFO) << ">>> NET BUILD";
	if (argc == 4) {
		LOG(INFO) << "Resuming from " << argv[3];
		solver.Solve(argv[3]);
	} else {
		solver.Solve(0);
	}
	LOG(INFO) << "Optimization Done.";
	//Net<float> train_net0(cnn_train);
	//Net<float> train_net1(cnn_train);

	/*
	   NetParameter trained_net_param;
	   caffe_test_net.ToProto(&trained_net_param, false);
	   WriteProtoToBinaryFile(trained_net_param, argv[2]);
	   */
	return 0;
}

