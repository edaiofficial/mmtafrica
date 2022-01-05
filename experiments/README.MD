# Running the experiments

Here is a guide on how to run the experiemnts. 

## Train MMTAFRICA

1. Create a virtual environment. See [here](https://docs.python.org/3/library/venv.html).
2. Activate your environment and install the packages in `requirements.txt`.
3. The file that runs the experiments is `mmt_bt.py`. Inside the file is the config:
  ```python
  HOME_PATH =  '/home/mila/c/chris.emezue/mmtall'
  if not os.path.exists(HOME_PATH):
      raise Exception(f'HOMEPATH {HOME_PATH} does not exist!')

  # Constants
  class Config():
      
    homepath = HOME_PATH
    prediction_path = os.path.join(homepath,'predictions/')
    # Use 'google/mt5-small' for non-pro cloab users
    model_repo = 'google/mt5-base'
    model_path_dir = HOME_PATH
    model_name = 'mmt_translation.pt'
    bt_data_dir = os.path.join(homepath,'btData')

    #Data part
    parallel_dir= os.path.join(HOME_PATH,'parallel')
    mono_dir= os.path.join(HOME_PATH,'mono')
    
    log = os.path.join(HOME_PATH,'train.log')
    mono_data_limit = 300
    mono_data_for_noise_limit=50
    #Training params
    n_epochs = 10
    n_bt_epochs=3

    batch_size = 64 #You can reduce this if you have low memory
    max_seq_len = 50
    min_seq_len = 2
    checkpoint_freq = 10000
    lr = 1e-4
    print_freq = 5000
    use_multiprocessing  = False

    num_cores = mp.cpu_count() 
    NUM_PRETRAIN = 20
    NUM_BACKTRANSLATION_TIMES = 5
    do_backtranslation=True
    now_on_bt=False
    bt_time=0
    using_reconstruction= True
    num_return_sequences_bt=2
    use_torch_data_parallel = True

    gradient_accumulation_batch = 4096//batch_size
    num_beams=4
  
    best_loss = 1000
    best_loss_delta = 0.00000001
    patience=15000000
    L2=0.0000001
    dropout=0.1
    
    drop_prob=0.2
    num_swaps=3
    
    verbose=True

    now_on_test=False
    
    #Initialization of state dict which will be saved during training
    state_dict = {'batch_idx': 0,'epoch':0,'bt_time':bt_time,'best_loss':best_loss}
    state_dict_check = {'batch_idx': 0,'epoch':0,'bt_time':bt_time,'best_loss':best_loss} #this is for tracing training after abrupt end!

    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
      
    #We will be leveraging parallel and monolingual data for each of these languages.
    #parallel data will be saved in a central 'parallel_data 'folder as 'src'_'tg'_parallel.tsv 
    #monolingual data will be saved in another folder called 'monolingual_data' as 'lg'_mono.tsv

    #Each tsv file is of the form "input", "output"
    LANG_TOKEN_MAPPING = {
      'ig': '<ig>',
      'fon': '<fon>',
      'en': '<en>',
      'fr': '<fr>',
      'rw':'<rw>',
      'yo':'<yo>',
      'xh':'<xh>',
      'sw':'<sw>'
    }
    
    truncation=True
  ```

  You only need to change `HOME_PATH` to your main directory where all the experiments are housed.
  The other parameters can be left as is to reproduce the results in our paper.

4. Run `python mmt_bt.py` after changing the `HOME_PATH` and other config variables as you want.


### Using sbatch

If you are sending an sbatch job, then look into the file `job.sh` and do the following.
1. Provide `YOUR_HOME_PATH` in lines 8, 9 and 14.
2. It is assumed you have already created the environment `env`.   
3. Run `sbatch job.sh`



## Evaluation

The file `evalmmt.py` is used to evaluate the model.