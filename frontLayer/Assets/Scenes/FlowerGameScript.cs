using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

public class FlowerGameScript : MonoBehaviour
{
    [SerializeField] public string cameraName = "Main Camera";

    private GameObject[] childObjects;

    // Start is called before the first frame update
    void Start()
    {
        int childCount = transform.childCount;
        childObjects = new GameObject[childCount - 1];
        int index = 0;
        for (int i = 0; i < childCount; i++)
        {
            Transform child = transform.GetChild(i);
            if (child.name != cameraName)
            {
                childObjects[index] = child.gameObject;
                index++;
            }
        }
        Debug.Log(childObjects.Length);
    }

    // Update is called once per frame
    void Update()
    {
        // Получаем позицию курсора мыши в мировых координатах
        //Vector3 mousePosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
        Vector3 mousePosition = Input.mousePosition;
        mousePosition.z = 0;


        //print(mousePosition);

        // Проверяем, находится ли курсор над одним из коллайдеров шестиугольников
        //Collider2D hitCollider = Physics2D.OverlapPoint(mousePosition);

        //if (hitCollider != null)
        //{
        //    // Определяем на каком из шестиугольников находится коллайдер
        //    foreach (GameObject hex in childObjects)
        //    {
        //        if (hitCollider.gameObject == hex)
        //        {
        //            Debug.Log("Cursor is over " + hex.name);
        //        }
        //    }
        //} else
        //{
        //    //Debug.Log("hitCollider is null");
        //}
    }
}
