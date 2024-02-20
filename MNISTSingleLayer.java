/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.quickstart.modeling.feedforward.classification;

import org.apache.commons.lang.time.StopWatch;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;


/**A Simple Multi Layered Perceptron (MLP) applied to digit classification for
 * the MNIST Dataset (http://yann.lecun.com/exdb/mnist/).
 *
 * This file builds one input layer and one hidden layer.
 *
 * The input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
 * (relu) activation function. The weights for this layer are initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer will have 1000 output signals to the hidden layer.
 *
 * The hidden layer has input dimensions of 1000. These are fed from the input layer. The weights
 * for this layer is also initialized using Xavier initialization. The activation function for this
 * layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
 * add up to 1. The highest of these normalized values is picked as the predicted class.
 *
 */
public class MNISTSingleLayer {

    private static Logger log = LoggerFactory.getLogger(MNISTSingleLayer.class);

    public static void main(String[] args) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 8; // number of epochs to perform

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(128)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        model.fit(mnistTrain, numEpochs);


        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(mnistTest);

        System.out.println("Start preparing test data");
        int samplesCount = 10000;

        DataSetIterator mnistTrain2 = new MnistDataSetIterator(1, samplesCount, false, false, true, rngSeed);
        //mnistTrain.reset();
        ArrayList<DataSet> datasets = new ArrayList<>();
        ArrayList<INDArray> dataSamples = new ArrayList<>();
        ArrayList<Integer> labelsInt = new ArrayList<>();
        int ii = 0;
        while(mnistTrain2.hasNext() && ii < samplesCount)
        {
            DataSet ds = mnistTrain2.next();
            INDArray labels = ds.getLabels();
            int label = getMaxValueFromINDArray(labels);
            labelsInt.add(label);
            dataSamples.add(ds.getFeatures());
            ii++;
        }

        int[] incorrect_correct_guesses = {0,0};
        System.out.println("ii: "+ii);

        StopWatch watch = new StopWatch();
        System.out.println("Begin Test");
        watch.start();
        for(int i =0; i <samplesCount; i++)
        {
            INDArray networkOutput = model.output(dataSamples.get(i) , false);
            int correspondingNumber = getMaxValueFromINDArray(networkOutput);
            (incorrect_correct_guesses[ (correspondingNumber == labelsInt.get(i))?1:0 ])++;
        }
        watch.stop();
        System.out.println("Test complete");
        System.out.println("Classify " + samplesCount + " samples, Time Elapsed: " + watch.getTime() + " ms" );
        System.out.println("Incorrect guesses: " + incorrect_correct_guesses[0] + ", incorect guesses fraction: " + (incorrect_correct_guesses[0]*1.0)/samplesCount);
        System.out.println("Correct guesses: " + incorrect_correct_guesses[1] + " correct guesses fraction: " + (incorrect_correct_guesses[1]*1.0)/samplesCount );
        //log.info(eval.stats());
        //log.info("****************Example finished********************");

    }

    private static int getMaxValueFromINDArray(INDArray array)
    {
        int retVal = 0;
        float biggest =0.0f;
        for(int i =0; i < array.size(1); i++)
        {
            if(array.getFloat(i) > biggest)
            {
                biggest = array.getFloat(i);
                retVal = i;
            }
        }
        return retVal;
    }

}
