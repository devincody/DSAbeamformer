class dada_handler{
private:
	multilog_t* log = 0;
	dada_hdu_t* hdu_in = 0;
	uint64_t header_size = 0;
	uint64_t block_size = 0;
	uint64_t bytes_read = 0;
	uint64_t block_id = 0;

	void dsaX_dbgpu_cleanup(void);
	int dada_cuda_dbregister (dada_hdu_t * hdu);
	int dada_cuda_dbunregister (dada_hdu_t * hdu);

public:
	dada_handler(char * name, int core, key_t in_key);
	~dada_handler();
	void read_headers(void);
	char* read();
	void close();
	bool check_transfers_complete();
	uint64_t get_block_size(){return block_size;}
	uint64_t get_bytes_read(){return bytes_read;}
};

dada_handler::dada_handler(char * name, int core, key_t in_key){
	log = multilog_open(name, 0);
	multilog_add(log, stderr);
	multilog(log, LOG_INFO, "creating hdu\n");

	hdu_in = dada_hdu_create(log);
	dada_hdu_set_key(hdu_in, in_key);

	if (dada_hdu_connect(hdu_in) < 0){
		printf ("Error: could not connect to dada buffer\n");
		exit(-1); // return EXIT_FAILURE;		
	}

	// lock read on buffer
	if (dada_hdu_lock_read (hdu_in) < 0) {
		printf ("Error: could not lock to dada buffer (try relaxing memlock limits in /etc/security/limits.conf)\n");
		exit(-1); // return EXIT_FAILURE;
	}

	if (dada_cuda_dbregister(hdu_in) < 0){
		printf ("Error: could not pin dada buffer\n");
		exit(-1); 
	}

	// Bind to cpu core
	if (core >= 0)
	{
		printf("binding to core %d\n", core);
		if (dada_bind_thread_to_core(core) < 0)
		printf("failed to bind to core %d\n", core);
	}

	#if VERBOSE
		multilog (log, LOG_INFO, "Done setting up buffer\n");
	#endif
}

dada_handler::~dada_handler(){
	dada_cuda_dbunregister(hdu_in);
}

void dada_handler::read_headers(){
	char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &(header_size));
	if (!header_in)
	{
		multilog(log ,LOG_ERR, "main: could not read next header\n");
		dsaX_dbgpu_cleanup();
		exit(-1);
	}

	if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
	{
		multilog (log, LOG_ERR, "could not mark header block cleared\n");
		dsaX_dbgpu_cleanup();
		exit(-1);
	}

	// size of block in dada buffer
	block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);

	#if VERBOSE
		multilog (log, LOG_INFO, "Done setting up header \n");
	#endif
	
	std::cout << "block size is: " << block_size << std::endl;
}

char* dada_handler::read(){
	return ipcio_open_block_read(hdu_in->data_block, &bytes_read, &block_id);
}

void dada_handler::close(){
	ipcio_close_block_read (hdu_in->data_block, bytes_read);
}

bool dada_handler::check_transfers_complete(){
	if (bytes_read != N_BYTES_PRE_EXPANSION_PER_BLOCK){
		std::cout << "ERROR: Async, Bytes Read: " << bytes_read << ", Should also be "<< N_BYTES_PRE_EXPANSION_PER_BLOCK << std::endl;
	}

	if (bytes_read < block_size){
		/* If there isn't enough data in the block, end the observation */
		#if VERBOSE
			std::cout <<"bytes_read < block_size, ending transfers" << std::endl;
		#endif
		// ipcio_close_block_read (hdu_in->data_block, bytes_read);
		return true;

	}
	
	return false;
}

void dada_handler::dsaX_dbgpu_cleanup() {
	/*cleanup as defined by dada example code */
	if (dada_hdu_unlock_read (hdu_in) < 0){
		multilog(log, LOG_ERR, "could not unlock read on hdu_in\n");
	}
	dada_hdu_destroy (hdu_in);
}


int dada_handler::dada_cuda_dbregister (dada_hdu_t * hdu) {
	/*! register the data_block in the hdu via cudaHostRegister */
	ipcbuf_t * db = (ipcbuf_t *) hdu->data_block;

	// ensure that the data blocks are SHM locked
	if (ipcbuf_lock (db) < 0) {
		perror("dada_dbregister: ipcbuf_lock failed\n");
		return -1;
	}

	// dont register buffers if they reside on the device
	if (ipcbuf_get_device(db) >= 0) {
		return 0;
	}
	size_t bufsz = db->sync->bufsz;
	unsigned int flags = 0;
	cudaError_t rval;
	// lock each data block buffer as cuda memory
	uint64_t ibuf;

	for (ibuf = 0; ibuf < db->sync->nbufs; ibuf++) {
		rval = cudaHostRegister ((void *) db->buffer[ibuf], bufsz, flags);
		if (rval != cudaSuccess) {
			perror("dada_dbregister:  cudaHostRegister failed\n");
			return -1;
		}
	}
	return 0;
}

int dada_handler::dada_cuda_dbunregister (dada_hdu_t * hdu) {
	/*! unregister the data_block in the hdu via cudaHostUnRegister */
	ipcbuf_t * db = (ipcbuf_t *) hdu->data_block;
	cudaError_t error_id;

	// dont unregister buffers if they reside on the device
	if (ipcbuf_get_device(db) >= 0)
	return 0;

	// lock each data block buffer as cuda memory
	uint64_t ibuf;
	for (ibuf = 0; ibuf < db->sync->nbufs; ibuf++) {
		error_id = cudaHostUnregister ((void *) db->buffer[ibuf]);
		if (error_id != cudaSuccess) {
			fprintf (stderr, "dada_dbunregister: cudaHostUnregister failed: %s\n",
			cudaGetErrorString(error_id));
	    	return -1;
	    }
	}
	return 0;
}