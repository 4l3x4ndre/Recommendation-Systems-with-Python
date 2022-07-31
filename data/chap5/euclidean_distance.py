import numpy as np


# Function to compute Euclidean Distance.
def euclidean(v1, v2):
    # Convert 1-D Python lists to numpy vectors
    v1 = np.array(v1)
    v2 = np.array(v2)
    # Compute vector which is the element wise square of the difference
    diff = np.power(np.array(v1)- np.array(v2), 2)
    # Perform summation of the elements of the above vector
    sigma_val = np.sum(diff)
    # Compute square root and return final Euclidean score
    euclid_score = np.sqrt(sigma_val)
    return euclid_score


# Define 3 users with ratings for 5 movies
u1 = [5,1,2,4,5]
u2 = [1,5,4,2,1]
u3 = [5,2,2,4,4]

print(euclidean(u1, u2))
print(euclidean(u1, u3))