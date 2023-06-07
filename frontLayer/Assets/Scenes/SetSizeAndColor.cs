using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;


public class SetSizeAndColor : MonoBehaviour
{

    [SerializeField] private TextMeshProUGUI valueText;

    // Start is called before the first frame update
    void Start()
    {
        getHexagonsSize();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void getHexagonsSize()
    {
        int percentSize = SliderScript.objectSizePercent;
        valueText.text = percentSize.ToString();
    }
}
