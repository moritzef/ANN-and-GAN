package ArtificialNeuralNetwork.Node;

public class TanhNode extends Node {

    public TanhNode(Node[] inputs, float[] weights, float bias) {
        super(inputs, weights, bias);
    }

    @Override
    public float function(){
        return (float) (-1+(2/(1+Math.exp(-2*calculateInput()))));
    }

    @Override
    public float calculateOutputFromDerivation(){
        float fn = function();
        return  1f-(fn*fn);
    }
}
