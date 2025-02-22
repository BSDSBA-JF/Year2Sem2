from matplotlib import pyplot as plt

# print(plt.style.available) 

plt.style.use('ggplot')

age_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]

# Note that lines are plotted by order of the line.

plt.plot(age_x, dev_y, label = "All Developers")

py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

plt.plot(age_x, py_dev_y, color = '#5a7d9a', label = "Python Developers")

js_dev_y = [37810, 43515, 46823, 49293, 53437,
            56373, 62375, 66674, 68745, 68746, 74583]

plt.plot(age_x, js_dev_y, label = 'JavaScript Developers')

plt.title('Median Salary in USD by Age')
plt.xlabel('Ages')
plt.ylabel('Median Salary in USD')

# This is one way but not recommended.
# plt.legend(['All Devs', 'Python'])

plt.legend()

# To add a grid
# plt.grid(True)

# Just to get some padding
# plt.tight_layout()

plt.show()