from csdp_datastore import ABC

raw_data_path = ""
output_data_path = ""
download_token = ""

a = ABC(dataset_path = raw_data_path, 
        output_path = output_data_path, 
        output_sample_rate = 128,
        download_token = download_token)

a.download() 
a.port_data()
