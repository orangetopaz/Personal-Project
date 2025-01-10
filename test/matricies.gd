extends Node2D

class_name Matricies

func matrix(rows: int, cols: int) -> Array:
	var out: Array = []
	for y in range(rows):
		out.append([])
		for x in range(cols):
			out[y].append(0.0)
	return out

func dot(m1: Array, m2: Array) -> float:
	var out
	for term in range(len(m1)):
		out += float(m1[term])*float(m2[term])
	return out

func matrix_multiply(a: Array, b: Array) -> Array:
	# Check if matrices can be multiplied
	if a.size() == 0 or b.size() == 0 or a[0].size() != b.size():
		push_error("Matrices cannot be multiplied. Number of columns in 'a' must equal the number of rows in 'b'.")
		return [null]
	
	
	var result: Array = matrix(a.size(), b[0].size())
	
	# Perform matrix multiplication
	for i in range(a.size()):
		for j in range(b[0].size()):
			for k in range(a[0].size()):
				result[i][j] += a[i][k] * b[k][j]
	
	return result

func matrix_add(a: Array, b: Array) -> Array:
	if a.size() == 0 or b.size() == 0 or a.size() != b.size() or a[0].size() != b[0].size():
		print(a.size())
		print(b.size())
		print(a[0].size())
		print(b[0].size())
		push_error("Matrices cannot be added. The dimensions of 'a' must equal that of 'b'.")
		return [null]
	var out = matrix(a.size(), a[0].size())
	for i in range(a.size()):
		for j in range(a[0].size()):
			out[i][j] = a[i][j]+b[i][j]
	return out

func mprint(mat: Array):
	for row in mat: print(row)
