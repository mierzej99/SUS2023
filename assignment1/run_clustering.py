import argparse

def arguments():
    """
    prepering program argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--file_path')
    return parser.parse_args()

def main():
    args = arguments()
    print(args)


if __name__ == '__main__':
    main()
