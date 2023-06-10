using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class MainHexagonScript : MonoBehaviour
{
    //private static string pathToFoodPictures = "Assets/Images/";
    private static string pathToFoodPictures = "Assets/Resources/";
    [SerializeField] public static string imageContainerName = "ImageHolder";

    public static bool correctFoodPictureIsSelected = false;

    [SerializeField] private TextMeshProUGUI correctcountText;

    public static string currentMainPictureName;
    public static GameObject hexagonWithCorrectPicture;
    public static CheckMouseScript hexagonScriptWithCorrectPicture;
    private Image imageComponent;

    private Transform imageContainerTransform;

    private string[] foodPictures = new string[9] {
        "Капуста",
        "Банан",
        "Клубника",
        "Виноград",
        "Лимон",
        "Яблоко",
        "Перец",
        "Апельсин",
        "Редька"
    };

    private string[] shuffledPicturesNameArray;

    [SerializeField] public static string hexagonsParentGameObjectName = "HexagonsParent";
    [SerializeField] private bool isDebug = false;

    private static System.Random random = new System.Random();

    private GameObject[] hexagonsChildren;

    public int correctCount = 0;

    // Start is called before the first frame update
    void Start()
    {
        if (correctcountText != null)
        {
            correctcountText.text = correctCount.ToString();
        }
        imageContainerTransform = transform.Find(imageContainerName);
        imageComponent = imageContainerTransform.GetComponent<Image>();

        shuffledPicturesNameArray = (string[])foodPictures.Clone();
        shufflePicturesName();

        setCurrentFoodImageMainHexagon(shuffledPicturesNameArray[0]);

        if (isDebug)
        {
            Debug.Log("Нужно найти картинку: " + currentMainPictureName);
        }

        // Получаем ссылку на Transform HexagonsParent
        Transform hexagonsParentTransform = GameObject.Find(hexagonsParentGameObjectName).transform;

        // Получаем количество дочерних элементов у HexagonsParent (родительского объекта)
        int hexagonsChildrenCount = hexagonsParentTransform.childCount;

        if (isDebug)
        {
            Debug.Log("количество дочерних элементов у HexagonsParent: " + hexagonsChildrenCount);
        }

        hexagonsChildren = new GameObject[hexagonsChildrenCount];

        // Проходим по всем дочерним объектам и получаем ссылки на них
        for (int i = 0; i < hexagonsChildrenCount; i++)
        {
            // Получаем ссылку на i-й дочерний объект родительского объекта
            Transform hexagonTransform = hexagonsParentTransform.GetChild(i);

            // Получаем ссылку на GameObject i-го дочернего объекта
            GameObject hexagonObject = hexagonTransform.gameObject;

            if (isDebug)
            {
                Debug.Log("Имя шестиугольника: " + hexagonObject.name);
            }

            setCurrentPictureForChildHexagon(hexagonObject, shuffledPicturesNameArray[i]);

            hexagonsChildren[i] = hexagonObject;
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (correctFoodPictureIsSelected)
        {
            correctCount++;
            if (correctcountText != null)
            {
                correctcountText.text = correctCount.ToString();
            }
            correctFoodPictureIsSelected = false;
            if (isDebug)
            {
                Debug.Log("Выбрана верная картинка: " + currentMainPictureName);
            }

            int index = random.Next(0, hexagonsChildren.Length);
            shufflePicturesName();
            setCurrentFoodImageMainHexagon(shuffledPicturesNameArray[index]);
           
            for (int i = 0; i < hexagonsChildren.Length; i++)
            {
                setCurrentPictureForChildHexagon(hexagonsChildren[i], shuffledPicturesNameArray[i]);
            }
            if (isDebug)
            {
                Debug.Log("Новая картинка для поиска: " + currentMainPictureName);
            }
        }
    }

    // Перемешиваем массив в случайном порядке
    private void shufflePicturesName()
    {
        // Перемешиваем массив, используя алгоритм Фишера-Йетса
        for (int i = 0; i < shuffledPicturesNameArray.Length; i++)
        {
            int j = random.Next(i, shuffledPicturesNameArray.Length);
            string temp = shuffledPicturesNameArray[i];
            shuffledPicturesNameArray[i] = shuffledPicturesNameArray[j];
            shuffledPicturesNameArray[j] = temp;
        }
    }

    private void setCurrentFoodImageMainHexagon(string foodPictureName)
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
                imageComponent.sprite = foodPictureSprite;
                currentMainPictureName = foodPictureName;
            }
        }
        else
        {
            Debug.Log("ImageComponent MainHaxagon is null");
        }
    }

    private void setCurrentPictureForChildHexagon(GameObject childHexagon, string pictureName)
    {
        CheckMouseScript checkMouseScript = childHexagon.GetComponent<CheckMouseScript>();
        checkMouseScript.setCurrentFoodImage(pictureName);

        if (isDebug)
        {
            Debug.Log(childHexagon.name + ": " + pictureName);
        }
    }
}