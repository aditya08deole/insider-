import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class WasteClassificationModel {
  private MultiLayerNetwork model;
  private DataNormalization scaler;
  private List<String> classLabels;
  
  public WasteClassificationModel(File modelFile, File classLabelsFile) throws IOException {
    this.model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
    this.scaler = new ImagePreProcessingScaler();
    this.classLabels = FileUtils.readLines(classLabelsFile, Charset.defaultCharset());
  }
  
  public String classifyWaste(File imageFile) throws IOException {
    Mat image = Imgcodecs.imread(imageFile.getAbsolutePath());
    Imgproc.resize(image, image, new Size(224, 224));
    INDArray input = Nd4j.create(image.reshape(1, 3, 224, 224));
    scaler.preProcess(input);
    INDArray predictions = model.output(input);
    int classIndex = Nd4j.argMax(predictions, 1).getInt(0, 0);
    return classLabels.get(classIndex);
  }
}