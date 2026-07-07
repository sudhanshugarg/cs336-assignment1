from multiprocessing import Process


def worker():
    print("hello")

def main():
    process_count = 4
    ps = []

    for i in range(process_count):
        process = Process(target=worker)
        process.start()
        ps.append(process)

    for p in ps:
        p.join()

    print("Done")




if __name__ == "__main__":
    main()
