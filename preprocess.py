from csdp_datastore import ABC

raw_data_path = "C:/Users/au588953"
output_data_path = "C:/Users/au588953"
download_token = "20590-LAtzyK3659d5NN-2SMbS"

a = ABC(dataset_path = raw_data_path, 
        output_path = output_data_path, 
        output_sample_rate = 128,
        download_token = download_token)
 
a.port_data(download_first=True)