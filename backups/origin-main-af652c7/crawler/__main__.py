import multiprocessing

from crawler.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
