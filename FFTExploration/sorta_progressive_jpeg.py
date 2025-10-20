@app.cell
def _(Generator, Tuple, cv2, np, os):
    def images_from_directory(dir: str)-> Generator[Tuple[str,np.ndarray]]:
        listing = os.listdir(dir)
        print(listing)
        for item in listing:
            path = os.path.join(dir, item)

            img = cv2.imread(path, 3)
            if img is None:
                print(f"Could not load {item} as image")
                continue
            yield item, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (images_from_directory,)

