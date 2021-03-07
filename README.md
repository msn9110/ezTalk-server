# Chinese Speech Recognition System (Using guide)

## Settings

`config/settings.json`

- **data_set_root** : specific where is your data set root such that all data is under this directory

- source (_optional_) : specific the root directory of the single pronounce directories (such as ㄨㄛˇ, ㄅㄚ, ...)

- sentence (*optional*) : specific the root directory of the all chinese word sentence (such as 我想吃飯)

- \_background_noise\_ (*optional*) : the folder that contains noises (*.wav)

- training\_set\_root (*optional*) : specific where is the root of the training set

- testing\_set (*optional*) : specific where is the testing set

- **clip_duration** (float type) : the most important duration of a wave file in seconds (suggest 0.8)

	 **NOTE** : optional item is default to data_set_root/(key)

  <br />
	 **model_settings** (will auto update after exec_train.py) : the default using model for this project

  <br />
	 **restful_settings** : fill your ip, port and uploads_dir then start the recognition restful service (will introduce

  how to start the service later)

  <br />
	 **training_set_settings** (using in prepare_dataset.py) :

	1. It will scan all wave files in source folder, and reduce their label to 'no tone' label
(e.g.:ㄋㄧˊ, ㄋㄧˇ are same as ㄋㄧ). Then, start to clip the important duration of all wave files.

	2. Scan the all wave files in sentence folder, and clip them into no tone label form. The file will be
discarded if the number of clip waves are not equal to the length of the sentence.

	3. merge previous 2 steps files according to 'no tone' label

	4. Shuffling the files for every 'no tone' label

	5. Compute how many wave files should take according to **testing_percentage** for wave files of a 'no tone' label
		>  * if the amount of the wave files of 'no tone' label, then the amount of the files must have at least **num_of_test** files for test
		>
		>  * else the amount of the testing files is the half of the amount for that class

	6. **using_all** attribute is used with **max_num_of_files**

	   > *  if **using_all** , the remained part of that class is using for training
	   >
	   > *  else, training set will use **max_num_of_files** for training, the remaining part will add to testing set

	7. start to prepare


###  How to Train Model :
1.  Auto Mode:
   	```
   	./train.sh or sh train.sh
   	```
	In this mode, it will automatically prepare_dataset -> (start to run in background) -> training 3 parts model by settings default values -> output *.pb -> testing, when run in background, you can use look_exec_train.sh to look detail.

	you can use `python3 get_current_state.py` to look  progress.
 if you want  to stop `python3 stop_executor.py`.

2. Semi Auto Mode:
	```
	python3 exec_train.py
	```
	you can set some arguments by command line using `-h, --help` to see detail. if you not set anything, default is settings.json, you can use `python3 get_current_state.py` to look  progress.
 if you want  to stop `python3 stop_executor.py`

3. Manual Mode:

	Use `python3 train_v1.py -h` to  see how to use

### How to Test Model Easily :

-	**The testing set is set in settings.json**

	if you want to test the latest model, just run the following command
   ```python3 exec_test.py```

	the other models, you can specify some arguments, for detail `-h`

	you can use `python3 get_current_state.py` to look  progress.
 if you want  to stop `python3 stop_executor.py`

- the result files are in the folder of *.pb


### Try to Test Making A Sentence with testing_set
for example,

```
python3 take_audio.py -s [要測試的句子]
```

### Start Recognition Service
start
```
./start_rest_speech_recognition.sh
```
stop
```
./stop_rest.sh
```
look log
```
./look_rest_log.sh
```
Ctrl-C to exit, but not stopped