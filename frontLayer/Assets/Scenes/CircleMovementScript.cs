using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class CircleMovementScript : MonoBehaviour
{
    public float speed = 3f;  // Скорость движения точки
    public float distance = 5f;  // Расстояние, которое точка пройдет влево и вправо от начальной позиции

    private Vector3 startPosition;
    private float maxX;
    private float minX;
    private bool movingRight = true;
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
        float halfWidth = parentRectTransform.rect.width / 2;
        minX = startPosition.x - distance - halfWidth;  // Вычисляем минимальную позицию по оси X
        maxX = startPosition.x + distance + halfWidth;  // Вычисляем максимальную позицию по оси X
    }

    void Update()
    {
        // Перемещение точки
        if (movingRight)
        {
            transform.localPosition += Vector3.right * speed * Time.deltaTime;
            if (transform.localPosition.x >= maxX)
                movingRight = false;
        }
        else
        {
            transform.localPosition += Vector3.left * speed * Time.deltaTime;
            if (transform.localPosition.x <= minX)
                movingRight = true;
        }
    }

    private void OnTriggerEnter2D(Collider2D collision)
    {
        myRenderer.material.color = Color.green;

    }

    private void OnTriggerExit2D(Collider2D collision)
    {
        myRenderer.material.color = Color.red;
    }
}
