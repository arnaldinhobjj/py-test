def relu(x):
    if x <= 0:
        return 0
    return x

n1_weight1 = 0
n1_weight2 = 0
n1_bias1 = 0

n2_weight1 = 0
n2_weight2 = 0
n2_bias2 = 0

learning_rate = 0.01

training_data = [
    (0, 0, 1),
    (1, 0, 3),
    (0, 1, 4),
    (1, 1, 6),
    (2, 1, 8),
    (1, 2, 9),
    (2, 2, 11)
]

for i in range(1000):
    for input1, input2, target in training_data:
        n1_weighted_sum = (input1 * n1_weight1) + (input2 * n1_weight2) + n1_bias1
        n1_output = relu(n1_weighted_sum)

        n1_error = target - n1_output

        n1_adjustment1 = n1_error * input1 * learning_rate
        n1_adjustment2 = n1_error * input2 * learning_rate

        n1_weight1 = n1_weight1 + n1_adjustment1
        n1_weight2 = n1_weight2 + n1_adjustment2

        n2_weighted_sum = (input1 * n2_weight1) + (input2 * n2_weight2) + n2_bias2
        n2_output = relu(n2_weighted_sum)

        n2_error = target - n2_output

        n2_adjustment1 = n2_error * input1 * learning_rate
        n2_adjustment2 = n2_error * input2 * learning_rate

        n2_weight1 = n2_weight1 + n2_adjustment1
        n2_weight2 = n2_weight2 + n2_adjustment2


        n1_bias1 = n1_bias1 + (n1_error * learning_rate)


print(f"Neuron 1 weight 1: {n1_weight1}")
print(f"Neuron 1 weight 2: {n1_weight2}")
print(f"Neuron 1 Bias: {n1_bias1}")
print(f"Neuron 1 Prediction: {n1_output}")
print(f"Neuron 1 Error: {n1_error}")

print(f"Neuron 2 weight 1: {n2_weight1}")
print(f"Neuron 2 weight 2: {n2_weight2}")
print(f"Neuron 2 Bias: {n2_bias2}")
print(f"Neuron 2 Prediction: {n2_output}")
print(f"Neuron 2 Error: {n2_error}")

