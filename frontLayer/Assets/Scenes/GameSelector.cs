using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;


public class GameSelector : MonoBehaviour
{
    // Start is called before the first frame update
    //void Start()
    //{
        
    //}

    //// Update is called once per frame
    //void Update()
    //{
        
    //}

    public void StartFlowerGame()
    {
        SceneManager.LoadScene("FlowerGame");
    }

    public void StartCrossesGame()
    {
        SceneManager.LoadScene("CrossesGame");
    }
}
