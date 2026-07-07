from multiprocessing import Process


def worker(x: str):
    print(f"hello {x}")

def main():
    process_count = 4
    ps = []

    names = ["sud", "uttam", "frank", "danny"]
    for i in range(process_count):
        process = Process(target=worker, args=(names[i],)) #to make it a tuple of size 1
        process.start()
        ps.append(process)

    for p in ps:
        p.join()

    print("Done")

if __name__ == "__main__":
    main()
