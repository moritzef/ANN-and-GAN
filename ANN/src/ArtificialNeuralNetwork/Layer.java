package ArtificialNeuralNetwork;
import ArtificialNeuralNetwork.Node.*;

public class Layer {
    public Node[] nodes;
    public Layer(int numberOfNodes, String nodeType,  Node[] previousNodes, float[][] previousWeights, float[] bias,boolean isInputLayer){
        nodes = new Node[numberOfNodes];
        for (int i = 0; i < numberOfNodes; i++) {
            if(!isInputLayer)
                switch (nodeType){
                    case "ReLU": {nodes[i] = new ReLUNode(previousNodes, previousWeights[i], bias[i]); break;}
                    case "Logistic": {nodes[i] = new LogisticNode(previousNodes, previousWeights[i], bias[i]); break;}
                    case "Swish": {nodes[i] = new SwishNode(previousNodes, previousWeights[i], bias[i]); break;}
                    case "Tanh": {nodes[i] = new TanhNode(previousNodes, previousWeights[i], bias[i]); break;}
                }
            else
                switch (nodeType){
                    case "ReLU": {nodes[i] = new ReLUNode(null, null,0); break;}
                    case "Logistic": {nodes[i] = new LogisticNode(null, null,0); break;}
                    case "Swish": {nodes[i] = new SwishNode(null, null,0); break;}
                    case "Tanh": {nodes[i] = new TanhNode(null, null,0); break;}
                }
        }
    }

    public void updateLayer(float[][] updateWeights, float[] updateBiases){
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].updateNode(updateWeights[i], updateBiases[i]);
        }
    }

    public void calculateLayerOutput(){
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].calculateOutput();
        }
    }

    public float[] getOutputArray(){
        float[] result = new  float[nodes.length];
        for (int i = 0; i < nodes.length; i++) {
            result[i] = nodes[i].output;
        }
        return result;
    }
}
