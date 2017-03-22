package Concept_detection;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.File;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.highgui.Highgui;

import com.sun.prism.image.Coords;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
//import java.io.File;
import java.util.LinkedList;
import java.util.List;

/**
 * 
 * Concept detection realization steps:
 * 1,Calculate the similarity between query image and candidate images
 * 2,Rank the candidate images
 * 3,Retrieve the distinctive visual element from an image by using SIFT
 * 4,Store the distinctive visual element as cloud
 *
 */

public class Concept_Detection {

	/**
	 * @throws IOException 
	 * @Yiran Liu
	 */
	public static void main(String[] args) throws IOException {
		// read the images
		String bookScene = "/home/yiran/Desktop/SIFT_test/SIFT_TEST/images/tablescene.jpg";
		File input = new File(bookScene);
		BufferedImage image = ImageIO.read(input); 
		int w = image.getWidth(null);
		int h = image.getHeight(null);
		System.out.println("Show the image" + image);
		
		
		candidateSelect("/home/yiran/Desktop/San_Francisco Dataset/00050000_00051000_3", "/home/yiran/Desktop/San_Francisco Dataset/QueryImg/PCI_sp_25163_37.802833_-122.428091_937772320_2_670882349_6.37781_18.0478.jpg" );
		
		
		//draw the images to check experiment results
		BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
		Graphics g = bi.getGraphics();
		g.drawImage(image, 0, 0,null);	

	}
	
	public static void candidateSelect(String folderPath, String QueryImgPath){
		
		//put for loop here! Select the candidate image from a folder
		int fileLength = new File(folderPath).list().length;
		
		File folder = new File(folderPath);
		File[] listOfFiles = folder.listFiles();
		String fileNameQuery = FilenameUtils.getName(QueryImgPath);
		int count = 0;
		System.out.println("Start selecting the candidate");
		for (int i =0;i<fileLength;i++){
	    	String str = listOfFiles[i].getName();
	    	double Dist = calDistance(str, fileNameQuery);     	
	    	if(Dist > 3000){
	    		int Simi = calSimilarity(folderPath + "/" + str, QueryImgPath);
	    		if(Simi > 3000){
	    			count++;
		    		System.out.println(listOfFiles[i]);
			    	File source = new File(folderPath + "/" + listOfFiles[i].getName());
			    	File dest = new File("/home/yiran/Desktop/San_Francisco Dataset/CandidateList" + "/" + listOfFiles[i].getName());
//			    	System.out.println(source);
			    	try{
			    		FileUtils.copyFile(source, dest);
			    	} catch (IOException e){
			    		e.printStackTrace();
			    	}
	    		}else{
	    			System.out.println("The threshold of similarity is too large!!!");
	    		}	
	    		System.out.println("Totally " + count + " fullfill the requirment");
	    	}else{
	    		System.out.println("The threshold of distance is too large!!!");
	    	}
	    }	
	}
	//calculate distances(meters) in one folder
	public static float calDistance(String CandidateImg, String QueryImg){
		//read locations in folder
    	String[] parts1 = CandidateImg.split("_");
    	float lat1 = Float.parseFloat(parts1[3]);
    	float lon1= Float.parseFloat(parts1[4]);	
    	String[] parts2 = QueryImg.split("_");
    	float lat2 = Float.parseFloat(parts2[3]);
    	float lon2= Float.parseFloat(parts2[4]);
	    double earthRadius = 6371000; //meters
	    double dLat = Math.toRadians(lat2-lat1);
	    double dLng = Math.toRadians(lon2-lon1);
	    double a = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) * Math.sin(dLng/2) * Math.sin(dLng/2);
	    double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
	    float dist = (float) (earthRadius * c);	
	    return dist; 
	}
	
	public static int calSimilarity(String Pathimg1, String Pathimg2){
		//calculate the similarity between two images
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat objectImage = Highgui.imread(Pathimg1, Highgui.CV_LOAD_IMAGE_COLOR);
        Mat sceneImage = Highgui.imread(Pathimg2, Highgui.CV_LOAD_IMAGE_COLOR);

        MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
        featureDetector.detect(objectImage, objectKeyPoints);
        KeyPoint[] keypoints = objectKeyPoints.toArray();
        
        MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();  
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        descriptorExtractor.compute(objectImage, objectKeyPoints, objectDescriptors);

        // Create the matrix for output image.
        Mat outputImage = new Mat(objectImage.rows(), objectImage.cols(), Highgui.CV_LOAD_IMAGE_COLOR);
        Scalar newKeypointColor = new Scalar(255, 0, 0);
        Features2d.drawKeypoints(objectImage, objectKeyPoints, outputImage, newKeypointColor, 0);

        // Match object image with the scene image
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();
        featureDetector.detect(sceneImage, sceneKeyPoints);
        descriptorExtractor.compute(sceneImage, sceneKeyPoints, sceneDescriptors);

        Mat matchoutput = new Mat(sceneImage.rows() * 2, sceneImage.cols() * 2, Highgui.CV_LOAD_IMAGE_COLOR);
        Scalar matchestColor = new Scalar(0, 255, 0);

        List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors, matches, 2);
        LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();
        return matches.size();		
	}
	public static void extractConcept(String Pathimg1, String Pathimg2){
		//this part can be used to segment the images(using Canny Edge Detection, extract the common area)
        Mat img1 = Highgui.imread(Pathimg1, Highgui.CV_LOAD_IMAGE_COLOR);
        Mat img2 = Highgui.imread(Pathimg2, Highgui.CV_LOAD_IMAGE_COLOR);
		
		
	}
}
