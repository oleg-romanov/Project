using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class TopAndBottomCircleMovement : MonoBehaviour
{
    public float speed = 3f;  // Скорость движения точки
    public float distance = 5f;  // Расстояние, которое точка пройдет влево и вправо от начальной позиции

    private Vector3 startPosition;
    private float maxY;
    private float minY;
    private bool movingTop = true;
    private GameObject parentObject;
    private RectTransform parentRectTransform;
    private SpriteRenderer myRenderer;

    void Start()
    {
        parentObject = transform.parent.gameObject;
        parentRectTransform = parentObject.GetComponent<RectTransform>();
        startPosition = transform.localPosition;
        myRenderer = gameObject.GetComponent<SpriteRenderer>();
        //float halfWidth = GetComponent<RectTransform>().rect.width / 2; // Получаем половину ширины точки
        float halfWidth = parentRectTransform.rect.height / 2;
        Debug.Log(halfWidth);
        minY = startPosition.x - distance - halfWidth;  // Вычисляем минимальную позицию по оси X
        maxY = startPosition.x + distance + halfWidth;  // Вычисляем максимальную позицию по оси X
    }

    void Update()
    {
        // Перемещение точки
        if (movingTop)
        {
            transform.localPosition += Vector3.up * speed * Time.deltaTime;
            if (transform.localPosition.y >= maxY)
                movingTop = false;
        }
        else
        {
            transform.localPosition += Vector3.down * speed * Time.deltaTime;
            if (transform.localPosition.y <= minY)
                movingTop = true;
        }
    }

    private void OnTriggerEnter2D(Collider2D collision)
    {
        myRenderer.material.color = Color.blue;

    }

    private void OnTriggerExit2D(Collider2D collision)
    {
        myRenderer.material.color = Color.red;
    }
}
