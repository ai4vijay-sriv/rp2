import matplotlib.pyplot as plt

# Sample data
x = list(range(10))
y = [i**2 for i in x]

# Create the plot
plt.plot(x, y)

# Add a title and labels
plt.title("Plot of y = x^2")
plt.xlabel("x")
plt.ylabel("y")

# Display the graph
plt.show()
