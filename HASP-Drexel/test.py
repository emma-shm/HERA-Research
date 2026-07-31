import zipfile

with zipfile.ZipFile("/Users/emmamartignoni/Downloads/TVAC_test_24thJuly.zip") as zf:
    print('hi')
    zf.printdir()