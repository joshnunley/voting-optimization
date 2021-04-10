from shutil import rmtree


def remove_folder(path):
    rmtree(path)
    print("Removal Complete")


folder = input("What is the folder name: ")
remove_folder(folder)