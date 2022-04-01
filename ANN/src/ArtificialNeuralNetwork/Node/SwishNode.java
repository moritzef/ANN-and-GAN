package ArtificialNeuralNetwork.Node;

public class SwishNode extends Node{

    public SwishNode(Node[] inputs, float[] weights, float bias) {
        super(inputs, weights, bias);
    }

    @Override
    public float function(){
        float inp = calculateInput();
        return (float) (inp/(1+Math.exp(-inp)));
    }

    @Override
    public float calculateOutputFromDerivation(){
        float fn = function();
        return (float) (fn+((1/(1+Math.exp(-calculateInput())))*(1-fn)));
    }
}
