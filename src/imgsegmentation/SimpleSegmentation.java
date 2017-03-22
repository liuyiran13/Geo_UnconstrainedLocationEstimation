package imgsegmentation;

import static marvin.MarvinPluginCollection.*;
import marvin.color.MarvinColorModelConverter;
import marvin.image.MarvinImage;
import marvin.image.MarvinSegment;
import marvin.io.MarvinImageIO;
import marvin.math.MarvinMath;

public class SimpleSegmentation {
    public SimpleSegmentation(){
        // 1. Load image
        MarvinImage original = MarvinImageIO.loadImage("/home/yiran/Desktop/SIFT_test/SIFT_TEST/images/tablescene.jpg");
        MarvinImage image = original.clone();
        System.out.println("Input image loaded: " + image);
        // 2. Change green pixels to white
        filterGreen(image);
        System.out.println("Green pixels changed to white");
        System.out.println("System is Drawing..........");
        // 3. Use threshold to separate foreground and background.
        MarvinImage bin = MarvinColorModelConverter.rgbToBinary(image, 127);
        // 4. Morphological closing to group separated parts of the same object
        morphologicalClosing(bin.clone(), bin, MarvinMath.getTrueMatrix(30, 30));
        // 5. Use Floodfill segmention to get image segments
        image = MarvinColorModelConverter.binaryToRgb(bin);
        MarvinSegment[] segments = floodfillSegmentation(image);
        System.out.println("Get image segments: " + segments);
        // 6. Show the segments in the original image
        for(int i=1; i<segments.length; i++){
            MarvinSegment seg = segments[i];
            System.out.println("The segmentation of query image is: " + seg);
            original.drawRect(seg.x1, seg.y1, seg.width, seg.height, java.awt.Color.yellow);
            //original.drawRect(seg.x1+1, seg.y1+1, seg.width, seg.height, java.awt.Color.yellow);
        }
        MarvinImageIO.saveImage(original, "/home/yiran/Desktop/SIFT_test/SIFT_TEST/images/tablescene_segmented.png");
        System.out.println("Drawing finished");
    }
    private void filterGreen(MarvinImage image){
        int r,g,b;
        for(int y=0; y<image.getHeight(); y++){
            for(int x=0; x<image.getWidth(); x++){
                r = image.getIntComponent0(x, y);
                g = image.getIntComponent1(x, y);
                b = image.getIntComponent2(x, y);
                if(r > g*0.9 && r > b*0.9){
                    image.setIntColor(x, y, 255,255,255);
        }}}
    }
    public static void main(String[] args) { new SimpleSegmentation();  }
}