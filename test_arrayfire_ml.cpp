//
// Created by inon on 9/10/18.
//

#ifndef OPEN_GPGPU_ANN_TEST_ARRAYFIRE_ML_H
#define OPEN_GPGPU_ANN_TEST_ARRAYFIRE_ML_H

#include <af/autograd.h>
#include <af/nn.h>
#include <af/optim.h>

#include <string>
#include <memory>

using namespace af;
using namespace af::nn;
using namespace af::autograd;

/**
 *
 * @param argc
 * @param args
 * @return
 *
 * @see "../arrayfire-ml/examples/xor.cpp"
 */
int xor_main(int argc, const char **args)
{
    std::string optimizer_arg = std::string(args[1]);

    const int inputSize  = 2;
    const int outputSize = 1;
    const double lr = 0.01;
    const double mu = 0.1;
    const int numSamples = 4;

    float hInput[] = {1, 1,
                      0, 0,
                      1, 0,
                      0, 1};

    float hOutput[] = {1,
                       0,
                       1,
                       1};

    auto in = af::array(inputSize, numSamples, hInput);
    auto out = af::array(outputSize, numSamples, hOutput);

    nn::Sequential model;

    model.add(nn::Linear(inputSize, outputSize));
    model.add(nn::Sigmoid());

    auto loss = nn::MeanSquaredError();

    std::unique_ptr<optim::Optimizer> optim;

    if (optimizer_arg == "--rmsprop") {
        optim = std::unique_ptr<optim::Optimizer>(new optim::RMSPropOptimizer(model.parameters(), lr));
    } else if (optimizer_arg == "--adam") {
        optim = std::unique_ptr<optim::Optimizer>(new optim::AdamOptimizer(model.parameters(), lr));
    } else {
        optim = std::unique_ptr<optim::Optimizer>(new optim::SGDOptimizer(model.parameters(), lr, mu));
    }

    Variable result, l;
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < numSamples; j++) {

            model.train();
            optim->zeroGrad();

            af::array in_j = in(af::span, j);
            af::array out_j = out(af::span, j);

            // Forward propagation
            result = model(nn::input(in_j));

            // Calculate loss
            l = loss(result, nn::noGrad(out_j));

            // Backward propagation
            l.backward();

            // Update parameters
            optim->update();
        }

        if ((i + 1) % 100 == 0) {
            model.eval();

            // Forward propagation
            result = model(nn::input(in));

            // Calculate loss
            // TODO: Use loss function (af::nn::MeanAbsoluteError public af::nn::Loss)
            af::array diff = out - result.array();
            printf("Average Error at iteration(%d) : %lf\n", i + 1, af::mean<float>(af::abs(diff)));
            printf("Predicted\n");
            af_print(result.array());
            printf("Expected\n");
            af_print(out);
            printf("\n\n");
        }
    }
    return 0;
}


/**
 *
 * @return
 *
 * @see https://github.com/plavin/arrayfire-ml/blob/alexnet/examples/alexnet.cpp
 */
int alexnet_main()
{
    const int inputSize  = 2;
    const int outputSize = 1;
    const double lr = 0.1;
    const int numSamples = 1;

    auto in = af::randu(227, 227, 3, numSamples);
    auto out = af::randu(55, 55, 96, 1);

    nn::Sequential alexnet;

    //alexnet.add(nn::Conv2D(11, 11, 4, 4, 0, 0, 3, 96, true));
    alexnet.add(nn::ReLU());

    Variable result;
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < numSamples; j++) {
            alexnet.train();
            // alexnet.zeroGrad();

            af::array in_j = in(af::span, af::span, af::span, j);
            af::array out_j = out;

            // Forward propagation
            result = alexnet.forward(nn::input(in_j));

            // Calculate loss
            // TODO: Use loss function (af::nn::MeanAbsoluteError public af::nn::Loss)
            af::array diff = out_j - result.array();

            // Backward propagation
            auto d_result = Variable(diff, false);
            result.backward(d_result);

            // Update parameters
            // TODO: Should use optimizer (af::optim::Optimizer)
            for (auto &param : alexnet.parameters()) {
                param.array() += lr * param.grad().array();
                param.array().eval();
            }
        }

        if ((i + 1) % 100 == 0) {
            alexnet.eval();

            // Forward propagation
            result = alexnet.forward(nn::input(in));

            // Calculate loss
            // TODO: Use loss function (af::nn::MeanAbsoluteError public af::nn::Loss)
            af::array diff = out - result.array();
            printf("Average Error at iteration(%d) : %lf\n", i + 1, af::mean<float>(af::abs(diff)));
            printf("Predicted\n");
            //af_print(result.array());
            printf("Expected\n");
            //af_print(out);
            printf("\n\n");
        }
    }
    return 0;
}

int main(int argc, const char** argv) {
    xor_main(argc, argv);
    alexnet_main();
}

#endif //OPEN_GPGPU_ANN_TEST_ARRAYFIRE_ML_H
