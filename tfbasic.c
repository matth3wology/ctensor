#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/c_api.h>

/*
 * Super basic example of using google tensorflow directly from C
 *
 */

// Using stack input data nothing to free
void tensor_free_none(void * data, size_t len, void* arg) {
}

TF_Operation * PlaceHolder(TF_Graph * graph, TF_Status * status, TF_DataType dtype, const char * name) {
	TF_OperationDescription * desc = TF_NewOperation(graph, "Placeholder", name);
	TF_SetAttrType(desc, "dtype", TF_FLOAT);
	return TF_FinishOperation(desc, status);
}

TF_Operation * Const(TF_Graph * graph, TF_Status * status, TF_Tensor * tensor, const char * name) {
	TF_OperationDescription * desc = TF_NewOperation(graph, "Const", name);
	TF_SetAttrTensor(desc, "value", tensor, status);
	TF_SetAttrType(desc, "dtype", TF_TensorType(tensor));
	return TF_FinishOperation(desc, status);
}

TF_Operation * Add(TF_Graph * graph, TF_Status * status, TF_Operation * one, TF_Operation * two, const char * name) {
	TF_OperationDescription * desc = TF_NewOperation(graph, "AddN", name);
	TF_Output add_inputs[2] = {{one, 0}, {two, 0}};
	TF_AddInputList(desc, add_inputs, 2);
	return TF_FinishOperation(desc, status);
}

int main() {
	printf("TensorFlow C library version: %s\n", TF_Version());

	TF_Graph * graph = TF_NewGraph();
	TF_SessionOptions * options = TF_NewSessionOptions();
	TF_Status * status = TF_NewStatus();
	TF_Session * session = TF_NewSession(graph, options, status);

	float in_val_one = 4.0f;
	float const_two = 2.0f;

	TF_Tensor * tensor_in = TF_NewTensor(TF_FLOAT, NULL, 0, &in_val_one, sizeof(float), tensor_free_none, NULL);
	TF_Tensor * tensor_out = NULL; // easy access after this is allocated by TF_SessionRun
	TF_Tensor * tensor_const_two = TF_NewTensor(TF_FLOAT, NULL, 0, &const_two, sizeof(float), tensor_free_none, NULL);

	// Operations
	TF_Operation * feed = PlaceHolder(graph, status, TF_FLOAT, "feed");
	TF_Operation * two = Const(graph, status, tensor_const_two, "const");
	TF_Operation * add = Add(graph, status, feed, two, "add");

	// Session Inputs
	TF_Output input_operations[] = { feed, 0 };
	TF_Tensor ** input_tensors = {&tensor_in};

	// Session Outputs
	TF_Output output_operations[] = { add, 0 };
	TF_Tensor ** output_tensors = {&tensor_out};

	TF_SessionRun(session, NULL,
			// Inputs
			input_operations, input_tensors, 1,
			// Outputs
			output_operations, output_tensors, 1,
			// Target operations
			NULL, 0, NULL,
			status);

	printf("Session Run Status: %d - %s\n", TF_GetCode(status), TF_Message(status) );
	printf("Output Tensor Type: %d\n", TF_TensorType(tensor_out));
	float * outval = TF_TensorData(tensor_out);
	printf("Output Tensor Value: %.2f\n", *outval);

	TF_CloseSession(session, status);
	TF_DeleteSession(session, status);

	TF_DeleteSessionOptions(options);

	TF_DeleteGraph(graph);

	TF_DeleteTensor(tensor_in);
	TF_DeleteTensor(tensor_out);
	TF_DeleteTensor(tensor_const_two);

	TF_DeleteStatus(status);
	return 0;
}
