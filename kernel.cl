__kernel void cnvs(__global float* fout, __global float* inputs, __global float* filter,
	int inDim, int outDim, int nbyn)
{
	int img = get_global_id(0);
	int Neuron = get_global_id(1);
	int loc = get_global_id(2);



	int fRow, fCol, floc, x, y, i;
	int offset = nbyn * nbyn;
	float sum = 0;


	sum = 0;

	for (fRow = 0; fRow < 3; fRow++)
	{
		for (fCol = 0; fCol < 3; fCol++)
		{
			x = loc % nbyn + fCol - 1;
			y = loc / nbyn + fRow - 1;
			if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) //zero padding
			{
				sum += inputs[img*inDim*offset+(Neuron%inDim) * offset + y * nbyn + x] * filter[Neuron * 9 + 3 * fRow + fCol];

			}

		}
	}


	fout[img * outDim * inDim  * offset + ((Neuron/inDim) * offset + loc) * inDim + (Neuron % inDim)] = sum;

}

__kernel void cnvSum(__global float* fout,__global float* sumOut, __local float* local_sum, 
	int n,__global float* bias,int offset,int outDim) {
	int img = get_global_id(0);
	int i = get_global_id(1);
	int j = get_local_id(1);

	size_t group_size = get_local_size(1);
	size_t half_size = group_size / 2;
	local_sum[j] = (i < n) ? fout[img*n+i] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	while (half_size > 0) {
		if (j < half_size) {
			local_sum[j] += local_sum[j + half_size];
			if ((half_size * 2) < group_size) {
				if (j == 0)local_sum[j] += local_sum[j + (group_size - 1)];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		group_size = half_size;
		half_size = group_size / 2;
	}
	
	if (j == 0) {
		sumOut[img*outDim*offset+get_group_id(1)] =((local_sum[0] + bias[get_group_id(1)/offset]) < 0)? 0 : local_sum[0] + bias[get_group_id(1)/offset];
	}

}
__kernel void pooling(__global float* input,__global float* output, int Dim,int nbyn) {
	int img = get_global_id(0);
	int i = get_global_id(1);
	int j = get_global_id(2);
	int offset = nbyn * nbyn;
	int nb = nbyn / 2;
	int onset = nb * nb;
	float max = input[img*Dim*offset+i * offset + j * 2+(j/nb)*nbyn];
	max = (max < input[img * Dim * offset + i * offset + j * 2 + (j / nb) * nbyn + 1]) ? input[img * Dim * offset + i * offset + j * 2 + (j / nb) * nbyn + 1] : max;
	max = (max < input[img * Dim * offset + i * offset + j * 2 + (j / nb) * nbyn + nbyn]) ? input[img * Dim * offset + i * offset + j * 2 + (j / nb) * nbyn + nbyn] : max;
	max = (max < input[img * Dim * offset + i * offset + j * 2 + (j / nb) * nbyn + nbyn+1]) ? input[img * Dim * offset + i * offset + j * 2 + (j / nb) * nbyn + nbyn+1] : max;

	output[img * Dim * onset + i*onset+j] = max;
}
__kernel void fc(__global float* input, __global float* output, __global float* weight, __global float* bias,
	int inDim, int outDim, __local float* local_sum) {
	int img = get_global_id(0);
	int i = get_global_id(1);
	int j = get_local_id(1);
	size_t group_size = get_local_size(1);
	size_t half_size = group_size / 2;
	local_sum[j] = input[img * inDim+i % inDim] * weight[i];
	barrier(CLK_LOCAL_MEM_FENCE);

	while (half_size > 0) {
		if (j < half_size) {
			local_sum[j] += local_sum[j + half_size];
			if ((half_size * 2) < group_size) {
				if (j == 0)local_sum[j] += local_sum[j + (group_size - 1)];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		group_size = half_size;
		half_size = group_size / 2;
	}
	if (j == 0) {
		output[img*outDim+get_group_id(1)] = ((local_sum[0] + bias[get_group_id(1)]) < 0) ? 0 : local_sum[0] + bias[get_group_id(1)];
	}
}