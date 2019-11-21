#include "eml_net.h"
static const float test_cnn_layer0_weights[15] = { 0.335099f, -0.327454f, -0.376279f, 0.248336f, -0.182299f, -0.507812f, -0.149365f, 0.365584f, 0.327495f, 0.052394f, -0.007393f, 0.408010f, 0.132442f, -0.293957f, 0.070711f };
static const float test_cnn_layer0_biases[5] = { -2.972165f, 4.952836f, -2.664029f, 0.979136f, 4.941312f };
static const float test_cnn_layer1_weights[30] = { -2.739278f, -3.847261f, 1.066007f, 1.560457f, 0.849560f, 3.737083f, -2.141241f, -0.320659f, 3.274637f, 1.829387f, 0.080510f, -3.453879f, 3.431261f, -3.557888f, -2.433614f, 2.581983f, 0.709567f, -2.562016f, 2.122726f, 2.095687f, 1.048052f, -3.862905f, -3.422662f, 1.757886f, -3.566486f, 5.002520f, -1.574579f, -5.876140f, 3.566519f, -4.628964f };
static const float test_cnn_layer1_biases[6] = { 0.684713f, 0.257139f, -0.315801f, -0.183628f, -0.903953f, 0.248110f };
static float test_cnn_buf1[6];
static float test_cnn_buf2[6];
static const EmlNetLayer test_cnn_layers[2] = { 
{ 5, 3, test_cnn_layer0_weights, test_cnn_layer0_biases, EmlNetActivationLogistic }, 
{ 6, 5, test_cnn_layer1_weights, test_cnn_layer1_biases, EmlNetActivationSoftmax } };
static EmlNet test_cnn = { 2, test_cnn_layers, test_cnn_buf1, test_cnn_buf2, 6 };