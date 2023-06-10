using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;

public class CircleMouseFollow : MonoBehaviour
{

    private TcpClient client;
    private NetworkStream stream;
    //private byte[] buffer = new byte[1024];
    private const int bufferSize = 9;
    private byte[] buffer;

    public Vector2 targetPosition; // Заданные координаты (x,y)

    public GameObject pointPrefab;
    public Transform canvasParent;
    private GameObject myPoint;
    private RectTransform myPointRectTransform;
    private SpriteRenderer mySpriteRenderer;
    public float xPosition = 0f;
    public float yPosition = 0f;


    float mainSpeed = 10.0f; //regular speed
    float shiftAdd = 250.0f; //multiplied by how long shift is held.  Basically running
    float maxShift = 1000.0f; //Maximum speed when holdin gshift
    float camSens = 0.05f; //How sensitive it with mouse
    private Vector3 lastMouse = new Vector3(255, 255, 255); //kind of in the middle of the screen, rather than at the top (play)
    private float totalRun = 1.0f;


    // Start is called before the first frame update
    void Start()
    {
        createNewPoint();
        //myPointRectTransform.anchoredPosition = new Vector2(xPosition, yPosition);
        // Подключение к серверу
        client = new TcpClient();

        client.Connect("localhost", 9097);

        buffer = new byte[bufferSize];

        stream = client.GetStream();


    }

    // Update is called once per frame
    void Update()
    {
        // Получение координат
        if (stream != null && stream.DataAvailable)
        {
            int bytesRead = stream.Read(buffer, 0, bufferSize);

            if (bytesRead > 0)
            {
                byte infoByte = buffer[0];
                float x = System.BitConverter.ToSingle(buffer, 1);
                float y = System.BitConverter.ToSingle(buffer, 5);

                
                //if (x < 720)
                //{
                //    x -= 100;
                //}

                //if (x < 43)
                //{
                    
                //    var value = 300 + Random.Range(20, 80);
                //    x -= value;
                //}

                Debug.Log($"x: {x}, y: {y})");

                myPointRectTransform.anchoredPosition = new Vector3(x - 720, y - 450, 0);
            }
        }
    }

    private void OnDisable()
    {
        stream?.Close();
        stream = null;
    }

    void createNewPoint()
    {
        myPoint = Instantiate(pointPrefab, canvasParent);
        myPoint.tag = "Circle1";
        myPointRectTransform = myPoint.GetComponent<RectTransform>();
        //mySpriteRenderer = myPoint.GetComponent<SpriteRenderer>();

        //if (mySpriteRenderer != null)
        //{
        //    CircleCollider2D circleCollider2D = myPoint.AddComponent<CircleCollider2D>();
        //    circleCollider2D.radius = mySpriteRenderer.bounds.size.x;
        //}
    }
}