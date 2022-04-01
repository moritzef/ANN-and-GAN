package ArtificialNeuralNetwork.Node;

public class LogisticNode extends Node{
    public LogisticNode(Node[] inputs, float[] weights, float bias) {
        super(inputs, weights, bias);
    }

    @Override
    public float function(){
        return (float) (1/(1+Math.exp(-calculateInput())));
    }

    @Override
    public float calculateOutputFromDerivation(){
        float fn = function();
        return  fn*(1-fn);
    }
}
