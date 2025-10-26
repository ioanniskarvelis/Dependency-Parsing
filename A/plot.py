import matplotlib.pyplot as plt

# Extract UAS and LAS scores from the log
uas_scores = [92.89, 93.34, 93.75, 93.55, 93.35]
las_scores = [91.05, 91.70, 92.07, 92.00, 91.83]

# Create a list of epochs for the x-axis
epochs = list(range(1, len(uas_scores) + 1))

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(epochs, uas_scores, marker='o', label='UAS')
plt.plot(epochs, las_scores, marker='o', label='LAS')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('UAS and LAS Scores')
plt.xticks(epochs)  # Set the x-axis ticks to the epoch numbers
plt.legend()
plt.show()