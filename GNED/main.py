from generation_main import generation_main
from classification_main import classification_main
from config import TASK


if __name__ == "__main__":

    if TASK == 'GENERATION':
        generation_main()

    elif TASK == 'CLASSIFICATION':
        classification_main()




