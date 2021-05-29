Instructions :

1) Create a virtual environment (Either Conda or Pip) and install the requirements using requirements.txt by running the command,

                              conda create --name <env> --file requirements.txt

2) To get all the genres that are available in the dataset use,

                              python cor_reader.py --show_genres

3) To train the bot from scratch with Cornell movie dialogues dataset using cross-entropy method,
                      
                              python train_crossent.py --cuda --data <genre> -n <name_of_save_folder>

or just use 'python train_crossent.py --help' to know about different arguments that can be given

Example : python train_crossent.py --cuda --data family -n crossent-family


4) To train using SCST method,

                               python train_scst.py --cuda --data <genre> -l <saved_crossentropy_model_file> -n <name_of_save_folder>

Example : python train_scst.py --cuda --data family -l saves/xe-family/epoch_040_0.853_0.185.dat -n sc-family


5) To test the model,

                              python use_model.py -m <saved_model_location> -s "<input_phrase>"

To make the bot interact with itself, --self argument can be used.
Example : python use_model.py -m saves/scst-horror/epoch_040_0.629_0.138.dat -s "hi" --self 10

 