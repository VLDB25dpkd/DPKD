combined_text = ""

for i in range(10):
    with open("./random/sst2/cluster" + str(i) + ".txt", "r") as file:
        content = file.read()
        combined_text += content
print("write")
with open("./random/sst2/combined.txt", "w") as combined_file:
    combined_file.write(combined_text)
