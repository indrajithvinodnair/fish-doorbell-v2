
import azure.ai.vision as visionsdk

def image_analysis_sample_analyze_with_custom_model(endpoint,key,custom_model_name,img_file):
    """
    Analyze image using a custom-trained model, use
    https://microsoftlearning.github.io/mslearn-ai-vision/Instructions/Exercises/02-image-classification.html
    for reference
    """

    service_options = visionsdk.VisionServiceOptions(endpoint,key)

    vision_source = visionsdk.VisionSource(filename=img_file)

    analysis_options = visionsdk.ImageAnalysisOptions()

    # Set your custom model name here
    analysis_options.model_name = custom_model_name

    image_analyzer = visionsdk.ImageAnalyzer(service_options, vision_source, analysis_options)

    result = image_analyzer.analyze()

    if result.reason == visionsdk.ImageAnalysisResultReason.ANALYZED:

        if result.custom_objects is not None:
            print(" Custom Objects:")
            for object in result.custom_objects:
                print("   '{}', {} Confidence: {:.4f}".format(object.name, object.bounding_box, object.confidence))

        if result.custom_tags is not None:
            print(" Custom Tags:")
            for tag in result.custom_tags:
                print("   '{}', Confidence {:.4f}".format(tag.name, tag.confidence))

    else:

        error_details = visionsdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        print("   Error reason: {}".format(error_details.reason))
        print("   Error code: {}".format(error_details.error_code))
        print("   Error message: {}".format(error_details.message))
        print(" Did you set the computer vision endpoint and key?")
    return result.custom_tags
        

