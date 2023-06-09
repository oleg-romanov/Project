using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CheckMouseScript : MonoBehaviour
{
    //private static string pathToFoodPictures = "Assets/Images/";
    private static string pathToFoodPictures = "Assets/Resources/";
    public string imageContainerName = "ImageHolder";

    [SerializeField] private bool isDebug = false;

    public string currentFoodPictureName;
    private Image imageComponent;

    private Transform childTransform;

    // Start is called before the first frame update
    void Start()
    {
        if (transform == null)
        {
            Debug.Log("transform is null");
            return;
        }
        childTransform = transform.Find(imageContainerName);
        if (childTransform != null)
        {
            imageComponent = childTransform.GetComponent<Image>();
            if (imageComponent != null)
            {
                if (isDebug)
                {
                    Debug.Log("Image в CheckMouseScript не равен null");
                }
                imageComponent.raycastTarget = false;
            }
            else
            {
                Debug.Log("Image в CheckMouseScript равен null");
            }
        }
        else
        {
            Debug.Log("childTransform = null в CheckMouseScript");
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnMouseEnter()
    {
        // Сравниваем значение в текущем шестиугольнике со значением в главном (центральном)
        // Также передается параметр чтобы сравнивать без учета регистра
        if (isDebug)
        {
            Debug.Log("Коснулись шестиугольника");
        }
        if (MainHexagonScript.currentMainPictureName != null)
        {
            if (currentFoodPictureName.Equals(MainHexagonScript.currentMainPictureName))
            {
                MainHexagonScript.correctFoodPictureIsSelected = true;
            }
        }
    }

    public void setCurrentFoodImage(string foodPictureName)
    {
        if (childTransform != null)
        {
            if (imageComponent != null)
            {
                if (isDebug)
                {
                    Debug.Log(pathToFoodPictures + foodPictureName + ".png");
                }
                Sprite foodPictureSprite = Resources.Load<Sprite>(foodPictureName);
                if (foodPictureSprite != null)
                {
                    currentFoodPictureName = foodPictureName;
                    imageComponent.sprite = foodPictureSprite;
                }
                else
                {
                    Debug.Log("foodPictureSprite у " + name + " равен null");
                }
            }
            else
            {
                Debug.Log("imageComponent у " + name + " равен null");
            }
        }
        else
        {
            Debug.Log("childTransform у " + name + " равен null");
        }
    }
}
