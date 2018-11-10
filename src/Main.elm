module Main exposing (..)

import Browser
import Http
import Url.Builder as Url
import Dict
import Element exposing (Element, el, text, row, alignRight, fill, width, height, rgb255, spacing, centerY, padding, none, px)
import Element.Input as Input
import Element.Background as Background
import Element.Border as Border
import Element.Font as Font
import Json.Decode as Decode exposing (Decoder, int, string, dict, list)
import Json.Decode.Pipeline exposing (required)
import Json.Encode as Encode


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        }



-- MODEL


type alias Model =
    { text : String }


type alias Docforia =
    { text : String
    , properties : Properties
    , edges : List Edges
    , nodes : List Nodes
    }


type alias Edges =
    { variants : List String
    , layer : String
    , properties : List Properties
    , connections : List Int
    }


type alias Nodes =
    { variants : List String
    , layer : String
    , properties : List Properties
    , ranges : List Int
    }


type alias Properties =
    Dict.Dict String String


init : () -> ( Model, Cmd Msg )
init _ =
    ( Model "", Cmd.none )



-- UPDATE


type Msg
    = NewDocforia (Result Http.Error Docforia)
    | EditedText String


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NewDocforia data ->
            ( model, Cmd.none )

        EditedText newText ->
            ( { model | text = newText }, Cmd.none )



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
    Sub.none



-- VIEW


view model =
    Element.layout []
        (body model)


body : Model -> Element Msg
body model =
    row [ width fill, spacing 30 ]
        [ textInput
        , resultView model
        ]


textInput : Element Msg
textInput =
    el
        [ width fill
        , Border.rounded 3
        , padding 30
        ]
        (Input.multiline
            [ height (px 600) ]
            { label = Input.labelHidden ""
            , onChange = EditedText
            , placeholder = Nothing
            , spellcheck = False
            , text = ""
            }
        )


resultView : Model -> Element msg
resultView model =
    el [ width fill ] none



-- HTTP


getCoreNLP : String -> Cmd Msg
getCoreNLP lang =
    Http.send NewDocforia (Http.post (vildeApi lang "corenlp_3.8.0") (Http.jsonBody (Encode.string "This is a test.")) docforiaDecoder)


vildeApi : String -> String -> String
vildeApi lang config =
    Url.crossOrigin "http://vilde.cs.lth.se:9000" [ lang, config, "api", "json" ] []


docforiaDecoder =
    Decode.field "DM10" docforiaHelper


docforiaHelper =
    Decode.succeed Docforia
        |> required "text" string
        |> required "properties" propertiesDecoder
        |> required "edges" (list edgesDecoder)
        |> required "nodes" (list nodesDecoder)


propertiesDecoder =
    dict string


edgesDecoder =
    Decode.succeed Edges
        |> required "variants" (list string)
        |> required "layer" string
        |> required "properties" (list propertiesDecoder)
        |> required "connections" (list int)


nodesDecoder =
    Decode.succeed Nodes
        |> required "variants" (list string)
        |> required "layer" string
        |> required "properties" (list propertiesDecoder)
        |> required "ranges" (list int)
