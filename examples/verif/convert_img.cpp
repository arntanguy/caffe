// Copyright Yuheng Chen 2013

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <cassert>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

#if 0
void convert_dataset(const char* image_filename, const char* label_filename,
        const char* db_filename) {
  // Open files
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  //std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename;
  //CHECK(label_file) << "Unable to open file " << label_file;
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  label_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
  image_file.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  label_file.read(reinterpret_cast<char*>(&num_labels), 4);
  num_labels = swap_endian(num_labels);
  CHECK_EQ(num_items, num_labels);
  image_file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  image_file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);

  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";

  char label;
  char* pixels = new char[rows * cols];
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(1);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int itemid = 0; itemid < num_items; ++itemid) {
    image_file.read(pixels, rows * cols);
    label_file.read(&label, 1);
    datum.set_data(pixels, rows*cols);
    datum.set_label(label);
    datum.SerializeToString(&value);
    snprintf(key, kMaxKeyLength, "%08d", itemid);
    db->Put(leveldb::WriteOptions(), std::string(key), value);
  }

  delete db;
  delete pixels;
}
#endif

static void read_name2id(const char *fn, std::map<std::string, int> &m){
	FILE *fi = fopen(fn, "rb");
	CHECK(fi != NULL);
	int n;
	fscanf(fi, "%d", &n);
	char name[256];
	m.clear();
	for(int i=0;i<n;i++){
		int id;
		fscanf(fi, "%s %d", name, &id);
		m[name] = id;
	}
	fclose(fi);
}
static void read_label(const char *fn, std::vector<int> &m){
	FILE *fi = fopen(fn, "rb");
	CHECK(fi != NULL);
	int n;
	fscanf(fi, "%d", &n);
	m.resize(n);
	for(int i=0;i<n;i++){
		int id;
		fscanf(fi, "%d", &id);
		m[i] = id;
	}
	fclose(fi);
}
static void convert(const char* input, const char *output, 
		int local, int color, int ph, int pw,
		const char *lablelist)
{
	//XXX
	int local_flip;
	std::vector<int> labels;
	CHECK(local == 0);

	read_label(lablelist, labels);

	FILE *fi = fopen(input, "rb");
	CHECK(fi != NULL);
	fseek(fi, 0, SEEK_END);
	long size = ftell(fi);
	rewind(fi);

	long elem_size = ph * pw * sizeof(double);
	if(color)
		elem_size *= 3;
	long nelem = size / elem_size;
	CHECK_EQ(nelem, labels.size());
  	LOG(INFO) << "ph,pw " << ph << ','<<pw ;

	// Open leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(
			options, output, &db);
	CHECK(status.ok()) << "Failed to open leveldb " <<output  
		<< ". Is it already existing?";

	char label;
	const int kMaxKeyLength = 10;
	char key[kMaxKeyLength];
	std::string value;

	int nf = elem_size/sizeof(double);
	double *_buf = new double[nf];
  	LOG(INFO) << "A total of " << nelem << " items.";

	for (int itemid = 0; itemid < nelem; ++itemid) {
		//image_file.read(pixels, rows * cols);
		//label_file.read(&label, 1);
		//datum.set_data(pixels, rows*cols);
		caffe::Datum datum;
		datum.set_channels(color ? 3: 1);
		datum.set_height(ph);
		datum.set_width(pw);

		size_t nread = fread(_buf, elem_size, 1, fi);
		assert(nread == 1);
		for(int i=0;i<nf;i++)
			datum.add_float_data((float)_buf[i]);
		//XXX
		datum.set_label(labels[itemid]);
		datum.SerializeToString(&value);
		snprintf(key, kMaxKeyLength, "%08d", itemid);
		db->Put(leveldb::WriteOptions(), std::string(key), value);
	}


	delete [] _buf;

	fclose(fi);
}

int main(int argc, char** argv) {
	if (argc != 8) {
		fprintf(stderr, "arg wrong\n");
		exit(1);
	} else {
		google::InitGoogleLogging(argv[0]);
		convert(argv[1], argv[2], 
				atoi(argv[3]), atoi(argv[4]), 
				atoi(argv[5]), atoi(argv[6]),
			argv[7]
		       );
	}
	return 0;
}

