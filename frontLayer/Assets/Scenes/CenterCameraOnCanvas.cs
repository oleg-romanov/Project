using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CenterCameraOnCanvas : MonoBehaviour
{

    public Camera mainCamera;
    public Canvas canvas;

    // Start is called before the first frame update
    void Start()
    {
        RectTransform canvasRectTransform = canvas.GetComponent<RectTransform>();
        Vector2 canvasSize = canvasRectTransform.sizeDelta;

        mainCamera.transform.position = new Vector3(canvasSize.x / 2f, canvasSize.y / 2f, mainCamera.transform.position.z);
    }
}
