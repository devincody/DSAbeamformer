class dada_handler{
	private:
		dada_hdu_t* hdu_in = 0;
		multilog_t* log;
		uint64_t header_size = 0;
		bool bound_to_core = false;

	public:
		void dada_handler(char name[] = "beam", key_t in_key = 0x0000dada;){};
		void ~dada_handler(void);

		uint64_t get_block_size(void);

		char * get_new_block(uint64_t * bytes_read, uint64_t * block_id);
		void close_block(uint64_t bytes_read);

		void multilog_write(char *text);

};

void dada_handler(char name[] = "beam", key_t in_key = 0x0000dada, int core = 0){
	log = multilog_open(name, 0);
	multilog_add(log, stderr);
	multilog(log, LOG_INFO, "creating hdu\n");

	// create dada hdu
	hdu_in	= dada_hdu_create (log);
	// set the input hdu key
	dada_hdu_set_key (hdu_in, in_key);

	// connect to dada buffer
	if (dada_hdu_connect (hdu_in) < 0) {
		printf ("could not connect to dada buffer\n");
		return EXIT_FAILURE;
	}

	// lock read on buffer
	if (dada_hdu_lock_read (hdu_in) < 0) {
		printf ("could not lock to dada buffer\n");
		return EXIT_FAILURE;
	}

	// Bind to cpu core
	if (core >= 0)
	{
		printf("binding to core %d\n", core);
		if (dada_bind_thread_to_core(core) < 0)
		printf("failed to bind to core %d\n", core);
	}

	/* read the headers from the input HDU and mark as cleared
	   will block until header is present in dada ring buffer */
	char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
	if (!header_in)
	{
		multilog(log ,LOG_ERR, "main: could not read next header\n");
		dsaX_dbgpu_cleanup (hdu_in, log);
		return EXIT_FAILURE;
	}

	if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
	{
		multilog (log, LOG_ERR, "could not mark header block cleared\n");
		dsaX_dbgpu_cleanup (hdu_in, log);
		return EXIT_FAILURE;
	}
}

void ~dada_handler(void){
	/*cleanup as defined by dada example code */
	if (dada_hdu_unlock_read (hdu_in) < 0){
		multilog(log, LOG_ERR, "could not unlock read on hdu_in\n");
	}
	dada_hdu_destroy (hdu_in);
}

uint64_t dada_handler::get_block_size(void){
	return ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
}


void dada_handler::multilog_write(char *text){
	multilog(log, LOG_INFO, text);
}

char * dada_handler::get_new_block(uint64_t * bytes_read, uint64_t * block_id){
	return ipcio_open_block_read (hdu_in->data_block, bytes_read, block_id);
}

void dada_handler::close_block(uint64_t bytes_read){
	/* Mark PSRDADA block as read */
	ipcio_close_block_read (hdu_in->data_block, bytes_read);
}



