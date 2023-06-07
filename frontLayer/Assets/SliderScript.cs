using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;


public class SliderScript : MonoBehaviour
{

    public static int objectSizePercent;

    [SerializeField] private Slider _slider;

    [SerializeField] private TextMeshProUGUI _sliderText;

    // Start is called before the first frame update
    void Start()
    {
        objectSizePercent = 50;

        _slider.onValueChanged.AddListener((value) => {
            _sliderText.text = value.ToString("0");
            objectSizePercent = (int) Mathf.Round(value);
        });
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
