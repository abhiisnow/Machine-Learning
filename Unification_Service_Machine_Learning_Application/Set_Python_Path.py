import os

def main():
    path = os.getcwd()
    set_python_column_matcher = "setx  path \"%PATH%;" + path + "\\Python37\""
    print(set_python_column_matcher)
    set_python_script_column_matcher = "setx  path \"%PATH%;" + path + "\\Python37\Scripts\""
    print(set_python_script_column_matcher)
    set_python_lib_column_matcher = "setx  path \"%PATH%;" + path + "\\Python37\Lib\""
    print(set_python_lib_column_matcher)

    os.system(set_python_column_matcher)
    os.system(set_python_script_column_matcher)
    os.system(set_python_lib_column_matcher)

if __name__ == '__main__':
    main()