extends Node2D

class_name Matricies

const e = 2.718281828459045
var rng = RandomNumberGenerator.new()
const LR: float = 0.03
const epoch_size: int = 32

func matrix(rows: int, cols: int) -> Array:
	var out: Array = []
	for y in range(rows):
		out.append([])
		for x in range(cols):
			out[y].append(0.0)
	return out

func transpose(m) -> Array:
	var out: Array = matrix(m[0].size(), m.size())
	for i in range(m[0].size()):
		for j in range(m.size()):
			out[i][j] = m[j][i]
	return out

func matrix_dupe_across(m: Array, wid: int = 2) -> Array:
	var out: Array = matrix(m.size(), 0)
	for i in range(m.size()):
		for j in range(wid): # concat
			out[i] += m[i]
	return out

func matrix_dupe_down(m: Array, heit: int = 2) -> Array:
	var out: Array
	for i in range(heit): out += m
	return out

func dot(m1: Array, m2: Array) -> float:
	var out
	for term in range(len(m1)):
		out += float(m1[term])*float(m2[term])
	return out

func scal(m: Array, sc: float) -> Array:
	var out = matrix(m.size(), m[0].size())
	for i in range(m.size()):
		for j in range(m[0].size()):
			out[i][j] = m[i][j]*sc
	return out

func sum(m: Array) -> float:
	var out: float = 0.0
	for i in range(len(m)): 
		for j in range(len(m[0])): 
			out += m[i][j]
	return out

func matrix_multiply(m1: Array, m2: Array) -> Array:
	# Check if matrices can be multiplied
	if m1.size() == 0 or m2.size() == 0 or m1[0].size() != m2.size():
		push_error("Matrices cannot be multiplied. Number of columns in 'm1' must equal the number of rows in 'm2'.")
		return [null]
	
	
	var result: Array = matrix(m1.size(), m2[0].size())
	
	# Perform matrix multiplication
	for i in range(m1.size()):
		for j in range(m2[0].size()):
			for k in range(m1[0].size()):
				result[i][j] += m1[i][k] * m2[k][j]
	
	return result

func hadamard_multiply(m1: Array, m2: Array) -> Array:
	if m1.size() != m2.size() or m1[0].size() != m2[0].size():
		push_error("Matrices cannot be added. The dimensions of 'm1' must equal that of 'm2'.")
		return [null]
	var out = matrix(m1.size(), m1[0].size())
	for i in range(m1.size()):
		for j in range(m1[0].size()):
			out[i][j] = m1[i][j]*m2[i][j]
	return out
	

func matrix_add(m1: Array, m2: Array) -> Array:
	if m1.size() == 0 or m2.size() == 0 or m1.size() != m2.size() or m1[0].size() != m2[0].size():
		push_error("Matrices cannot be added. The dimensions of 'm1' must equal that of 'm2'.")
		return [null]
	var out = matrix(m1.size(), m1[0].size())
	for i in range(m1.size()):
		for j in range(m1[0].size()):
			out[i][j] = m1[i][j]+m2[i][j]
	return out

func matrix_sub(m1: Array, m2: Array) -> Array:
	if m1.size() == 0 or m2.size() == 0 or m1.size() != m2.size() or m1[0].size() != m2[0].size():
		push_error("Matrices cannot be added. The dimensions of 'm1' must equal that of 'm2'.")
		return [null]
	var out = matrix(m1.size(), m1[0].size())
	for i in range(m1.size()):
		for j in range(m1[0].size()):
			out[i][j] = m1[i][j]-m2[i][j]
	return out

func mprint(mat: Array):
	for row in mat: print(row)

func get_column(m: Array, index: int) -> Array:
	var column: Array = []
	for row in m:
		if index < row.size():
			column.append(row[index])
		else:
			print("Column index out of range for a row")
	return column


func snapped_matrix(m: Array, nearest: float) -> Array:
	var out: Array = matrix(m.size(), m[0].size())
	for row in range(len(m)):
		for col in range(m[row]):
			out[row][col] = snapped(m[row][col], nearest)
	return out

func LReLU(x: float) -> float:
	return max(0.05*x, x)

func dLReLU(x: float) -> float:
	if x > 0:
		return 1
	else:
		return 0.05

func σ(x: float) -> float:
	return 1/(1+e**(-0.03*x))

func dσ(x: float) -> float:
	var part_sig: float = e**(-0.3*x)
	return (0.3*part_sig)/(1+part_sig)**2

func logBase(x: float, base: float) -> float:
	return log(x)/log(base)

func ln(x: float) -> float:
	return logBase(x, e)

func matrix_logBase(m: Array, base: float) -> Array:
	var out: Array = matrix(m.size(), m[0].size())
	for i in range(m.size()): 
		for j in range(m[0].size()): 
			out[i][j] = logBase(m[i][j], base)
	return out

func matrix_ln(m: Array) -> Array:
	var out: Array = matrix(m.size(), m[0].size())
	for i in range(m.size()): 
		for j in range(m[0].size()): 
			out[i][j] = ln(m[i][j])
	return out

func matrix_σ(m: Array) -> Array:
	var out: Array = matrix(m.size(), m[0].size())
	for i in range(m.size()): 
		for j in range(m[0].size()): 
			out[i][j] = σ(m[i][j])
	return out

func matrix_dσ(m: Array) -> Array:
	var out: Array = matrix(m.size(), m[0].size())
	for i in range(m.size()): 
		for j in range(m[0].size()): 
			out[i][j] = dσ(m[i][j])
	return out
