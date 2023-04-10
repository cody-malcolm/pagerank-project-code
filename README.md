# PageRank Algorithm Project Code

## Contributors
Avril Lopez van Domselaar - 100746008

Cody Malcolm - 100753739

## Usage

### Running the Project

To run the project, execute the following command:

`python3 pagerank.py <settings_file> <h_matrix_file> [initial_x_file]`

The settings file is a text file that contains zero or more `key=value` pairs, as described in the next section. The matrix file is a `.csv` format file that describes the hyperlink matrix to be used, as described below. The initial x file, if used, is a `.csv` format file that describes the initial x vector to be used, as described below.

The program checks that the correct number of arguments are present, but does not validate that the given inputs in the files are correct or meaningful. It is up to the user to define their inputs correctly according to the instructions below.

### Settings Options

The settings file can contain 0 to 9 properties. All properties have default values and are therefore optional, however even if no properties are defined by the user, a (in this case empty) settings file still needs to be provided. Only one property can be listed on each line, and they can be listed in any order. A sample settings file with all keys set can be found in the `examples` directory.

The keys and their descriptions are as follows:
- `iterative`: Whether or not to perform the direct iteration method.
- `power`: Whether or not to perform the power iteration method.
- `eigenvector`: Whether or not to perform the dominant eigenvector method.
- `apply_random_surfer`: Whether or not to apply the random surfer algorithm to the given matrix.
- `apply_probability_normalization`: Whether or not to apply probability normalization after each iteration.
- `plot_results`: Whether or not to display a plot the results. Only works with the iterative method.
- `precision`: The number of decimal places to round final results to.
- `k`: The number of iterations to perform, for direct iteration and power iteration methods.
- `res`: The desired residual to use as a stopping criteria. Only applies to the direct iteration method. If provided, takes precedence over `k` for the direct iteration method.

The key name, expected/valid values, and default values are provided here:

key | valid inputs | default value 
----|--------------|--------------
iterative | `True` or `False` | `True`
power | `True` or `False` | `False`
eigenvector | `True` or `False` | `True`
apply_random_surfer | `True` or False | `False`
apply_probability_normalization | `True` or `False` | `True`
plot_results | `True` or `False` | `False`
precision | int >= 1 | 2
k | int >= 1 | 100
res | see below | 0.0001 if k not set, `None` if k set

There are four valid options for `res`:
- a float that is a meaningful residual, such as `0.00004`
- a string formatted like `10^-k`, where k is some positive integer
- a string formatted like `ax10^-k`, where a and k are positive integers
- the string `None`

### Matrix File Format

The matrix file expects a *modified* hyperlink matrix in `.csv` format where each line of the file corresponds to a row of the matrix. As with a typical hyperlink matrix (as described in the report), each column represents one page and row *i* of column *j* indicates whether or not page *j* contains an out link to page *i*. However, the value to use here is simply `1` instead of `1/|Pi|`. Thus the hyperlink matrix will be comprised only of `0`s and `1`s. The program will apply the random surfer algorithm (if requested), and then perform probability normalization on each column of the matrix to make it stochastic. This approach was taken to simplify the input process, simplify the code when random surfer is applied, and reduce instances of floating point imprecision. There are several example files in the `examples` directory.

### Initial X File Format

The initial x file expects a single row of input in `.csv` format. It expects a probability vector and will perform probability normalization if required. Since the user may not realize their input is not a probability vector, a warning will be printed in this case. An error will be thrown if the length of `x` does not match the size of `H`. There are several valid example files in the `examples` directory.

### Dependencies

In addition to default libraries, this project requires `numpy` and `matplotlib` which may need to be manually installed in order to run the program.


