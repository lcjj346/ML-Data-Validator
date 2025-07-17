import pandas as pd
import random
import faker

fake = faker.Faker()

def generate_data(n=200):
    data = []
    for _ in range(n):
        name = fake.name()
        email = fake.email()
        if random.random() < 0.2:
            email = email.replace("@", "_at_")  # introduce error

        age = random.randint(18, 60)
        if random.random() < 0.2:
            age = None  # missing value

        phone = fake.phone_number().replace("-", "")[:8]
        if random.random() < 0.2:
            phone = phone[:4] + "abc"  # error

        data.append({"Name": name, "Email": email, "Age": age, "Phone": phone})

    df = pd.DataFrame(data)
    df.to_excel("data/generated_data.xlsx", index=False)
    print("Generated synthetic data.")

if __name__ == "__main__":
    generate_data()
