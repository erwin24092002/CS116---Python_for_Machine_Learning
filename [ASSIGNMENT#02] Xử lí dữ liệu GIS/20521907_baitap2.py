import geopandas 

path = "CSL_HCMC/Data/GIS/Population/population_HCMC/population_shapefile/Population_Ward_Level.shp"
data = geopandas.read_file(path)
'''[Com_Name, Dist_Name, Com_ID, Dist_ID, Level, Pop_2009, Pop_2019, Den_2009, Den_2019, Shape_Leng, Shape_Area, geometry]'''

index = data["Shape_Area"].argmax()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có diện tích lớn nhất là {Com_Name}, {Dist_Name}")

index = data["Pop_2019"].argmax()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có dân số 2019 (Pop_2019) lớn nhất là {Com_Name}, {Dist_Name} ")

index = data["Shape_Area"].argmin()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có diện tích nhỏ nhất là {Com_Name}, {Dist_Name} ")

index = data["Pop_2019"].argmin()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có dân số thấp nhất (2019) là {Com_Name}, {Dist_Name} ")

index = ((data["Pop_2019"]-data["Pop_2009"])/data["Pop_2009"]).argmax()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có tốc độ tăng trưởng dân số nhanh nhất (dựa trên Pop_2009 và Pop_2019) là {Com_Name}, {Dist_Name} ")

index = ((data["Pop_2019"]-data["Pop_2009"])/data["Pop_2009"]).argmin()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có tốc độ tăng trưởng dân số thấp nhất là {Com_Name}, {Dist_Name} ")

index = (data["Pop_2019"]-data["Pop_2009"]).argmax()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có biến động dân số nhanh nhất là {Com_Name}, {Dist_Name} ")

index = (data["Pop_2019"]-data["Pop_2009"]).argmin()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có biến động dân số chậm nhất là {Com_Name}, {Dist_Name} ")

index = (data["Pop_2019"]/data["Shape_Area"]).argmax()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có mật độ dân số cao nhất (2019) là {Com_Name}, {Dist_Name} ")

index = (data["Pop_2019"]/data["Shape_Area"]).argmin()
Com_Name, Dist_Name = data["Com_Name"][index], data["Dist_Name"][index]
print(f"Phường có mật độ dân số thấp nhất (2019) là {Com_Name}, {Dist_Name} ")