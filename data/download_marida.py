import ee

def main():
    lc = ee.ImageCollection('MODIS/006/MCD12Q1')
    ee.Initialize()
    print("test")

if __name__ == '__main__':
    main()