from GNED.generation_main import generation_main
from GNED.classification_main import classification_main
from GNED.config import TASK


if __name__ == "__main__":

    if TASK == 'GENERATION':
        generation_main()

    elif TASK == 'CLASSIFICATION':
        classification_main()

    else:
        print('Please make sure you have selected the proper task.')




