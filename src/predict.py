from linearRegression import LinearRegression


def main():
    try:
        print ("enter a mileage: ")
        mileage = input()
        model = LinearRegression()
        model.load_weights()
        result = model.predict_for_input(model.weights, int(mileage))
        print(f"The estimated price is {result}")
    except Exception as e:
        print(f"An error occured: {e}")


if __name__ == "__main__":
    main()
